import csv
from os import error
import numpy as np
import pinocchio


class multicopterData():
    def __init__(self, file_name, prop_min_speed, prop_max_speed):
        self.file_name = file_name
        self.time_name = 'timestamp'

        self.prop_min_speed = prop_min_speed
        self.prop_max_speed = prop_max_speed

    def loadNumericalData(self, name_list, name_timestamp):
        # Angular acceleration
        values = []
        values_ts = []
        with open(self.file_name) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            for idx, row in enumerate(csv_reader):
                if idx > 0:
                    value = np.array([float(row[name]) for name in name_list if row[name] != ''])
                    if value.size > 0:
                        values.append(value)
                        values_ts.append(row[name_timestamp])
        return values, values_ts

    def loadLinearVelocity(self):
        # Angular acceleration
        rotor_names = [
            '/fmu/vehicle_local_position_groundtruth/out/vx', '/fmu/vehicle_local_position_groundtruth/out/vy',
            '/fmu/vehicle_local_position_groundtruth/out/vz'
        ]
        vels, vel_ts = self.loadNumericalData(rotor_names, '/fmu/vehicle_local_position_groundtruth/out/timestamp')

        return [fromFRDtoFLU(vel) for vel in vels], vel_ts

    def loadRotorAngularVelocity(self):
        # Angular acceleration
        rotor_names = [
            '/fmu/actuator_outputs/out/output.0',
            '/fmu/actuator_outputs/out/output.1',
            '/fmu/actuator_outputs/out/output.2',
            '/fmu/actuator_outputs/out/output.3',
        ]
        rotor_pwms, rotor_ts = self.loadNumericalData(rotor_names, '/fmu/actuator_outputs/out/timestamp')

        return [fromPWMtoRadS(rotor_pwm, self.prop_min_speed, self.prop_max_speed)
                for rotor_pwm in rotor_pwms], rotor_ts

    def loadAcceleration(self):
        # Accelerations
        acc_names = [
            '/fmu/sensor_combined/out/accelerometer_m_s2.0', '/fmu/sensor_combined/out/accelerometer_m_s2.1',
            '/fmu/sensor_combined/out/accelerometer_m_s2.2'
        ]
        acc_frds, acc_ts = self.loadNumericalData(acc_names, '/fmu/sensor_combined/out/timestamp')

        # Gravity removed from IMU's readings!
        return [fromFRDtoFLU(acc) + np.array([0, 0, -9.81]) for acc in acc_frds], acc_ts

    def loadAngularVelocity(self):
        # Angular velocity
        ang_vel_names = [
            '/fmu/sensor_combined/out/gyro_rad.0', '/fmu/sensor_combined/out/gyro_rad.1',
            '/fmu/sensor_combined/out/gyro_rad.2'
        ]
        ang_vel_frds, ang_vel_ts = self.loadNumericalData(ang_vel_names, '/fmu/sensor_combined/out/timestamp')

        return [fromFRDtoFLU(ang_vel) for ang_vel in ang_vel_frds], ang_vel_ts

    def loadAngularAcceleration(self):
        # Angular velocity
        ang_acc_names = [
            '/fmu/vehicle_angular_acceleration/out/xyz.0', '/fmu/vehicle_angular_acceleration/out/xyz.1',
            '/fmu/vehicle_angular_acceleration/out/xyz.2'
        ]
        ang_acc_frds, ang_acc_ts = self.loadNumericalData(ang_acc_names,
                                                          '/fmu/vehicle_angular_acceleration/out/timestamp')

        return [fromFRDtoFLU(ang_acc) for ang_acc in ang_acc_frds], ang_acc_ts

    def loadAttitude(self):
        quat_names = [
            '/fmu/vehicle_attitude_groundtruth/out/q.0', '/fmu/vehicle_attitude_groundtruth/out/q.1',
            '/fmu/vehicle_attitude_groundtruth/out/q.2', '/fmu/vehicle_attitude_groundtruth/out/q.3'
        ]

        quat_ned_frds, quat_ts = self.loadNumericalData(quat_names, '/fmu/vehicle_attitude_groundtruth/out/timestamp')

        R_nwu_flus = []
        R_nwu_ned = np.identity(3)
        R_nwu_ned[1, 1] = -1
        R_nwu_ned[2, 2] = -1
        R_frd_flu = R_nwu_ned

        for q in quat_ned_frds:
            q_ned_fdr = pinocchio.Quaternion(q[0], q[1], q[2], q[3])
            R_nwu_flu = R_nwu_ned @ q_ned_fdr.toRotationMatrix() @ R_frd_flu
            R_nwu_flus.append(R_nwu_flu)

        return R_nwu_flus, quat_ts


def fromFRDtoFLU(vec_frd):
    R_flu_frd = np.identity(3)
    R_flu_frd[1, 1] = -1
    R_flu_frd[2, 2] = -1

    return R_flu_frd @ vec_frd


def fromPWMtoRadS(pwm, prop_min, prop_max):
    pwm_max = 2000
    pwm_min = 1000
    ang_vel = prop_min + (pwm - pwm_min) / (pwm_max - pwm_min) * (prop_max - prop_min)
    return ang_vel


def computeD(acc, R, ang_vel, ang_acc):
    wx = ang_vel[0]
    wy = ang_vel[1]
    wz = ang_vel[2]

    wx_dot = ang_acc[0]
    wy_dot = ang_acc[1]
    wz_dot = ang_acc[2]

    R31 = R[2, 0]
    R32 = R[2, 1]
    R33 = R[2, 2]

    ax = acc[0]
    ay = acc[1]
    az = acc[2]

    g = 9.81

    r1 = np.array([-wy**2 - wz**2, wx * wy - wz_dot, wx * wz + wy_dot, 0, 0, 0, 0, 0, 0])
    r2 = np.array([wx * wy + wz_dot, -wx**2 - wz**2, -wx_dot + wy * wz, 0, 0, 0, 0, 0, 0])
    r3 = np.array([wx * wz - wy_dot, wx_dot + wy * wz, -wx**2 - wy**2, 0, 0, 0, 0, 0, 0])
    r4 = np.array([
        0, az + R33 * g, -ay - R32 * g, wx_dot, -wy * wz, wy * wz, -wx * wz + wy_dot, wx * wy + wz_dot, wy**2 - wz**2
    ])
    r5 = np.array([
        -az - R33 * g, 0, ax + R31 * g, wx * wz, wy_dot, -wx * wz, wx_dot + wy * wz, -wx**2 + wz**2, -wx * wy + wz_dot
    ])
    r6 = np.array(
        [ay + R32 * g, -ax - R31 * g, 0, -wx * wy, wx * wy, wz_dot, wx**2 - wy**2, wx_dot - wy * wz, wx * wz + wy_dot])

    return np.array([r1, r2, r3, r4, r5, r6])


def computeDm(acc, R):
    R31 = R[2, 0]
    R32 = R[2, 1]
    R33 = R[2, 2]

    ax = acc[0]
    ay = acc[1]
    az = acc[2]

    g = 9.81

    return np.array([ax + R31 * g, ay + R32 * g, az + R33 * g, 0, 0, 0])


def computeDDm(R, vel, acc):
    regressor = pinocchio.bodyRegressor(vel, acc)

    Dm = regressor[:, 0]
    Dm[:3] -= R.T @ np.array([0, 0, -9.81])

    D = np.vstack([
        regressor[:, 1], regressor[:, 2], regressor[:, 3], regressor[:, 4], regressor[:, 6], regressor[:, 9],
        regressor[:, 5], regressor[:, 7], regressor[:, 8]
    ]).T

    return D, Dm


def computeDk(rotors):
    n1 = rotors[0]
    n2 = rotors[1]
    n3 = rotors[2]
    n4 = rotors[3]

    r1 = [0.13, 0.22]
    r2 = [0.13, 0.2]
    r3 = [0.13, 0.22]
    r4 = [0.13, 0.2]

    mx = -r1[1] * n1**2 - r4[1] * n4**2 + r2[1] * n2**2 + r3[1] * n3**2
    my = -r1[0] * n1**2 - r3[0] * n3**2 + r2[0] * n2**2 + r4[0] * n4**2
    c1 = np.array([0, 0, n1**2 + n2**2 + n3**2 + n4**2, mx, my, 0])
    c2 = np.array([0, 0, 0, 0, 0, -n1**2 - n2**2 + n3**2 + n4**2])

    return np.array([c1, c2]).T

def printResults(params, Xt, Sigma):
    if len(params) != Xt.size - 1:
        print("Wrong parameter vector dimension")

    print("\n\nParameter values:")
    for idx, param in enumerate(params):
        print("Param", param, ":", Xt[idx], ". St. dev.:", Sigma[idx])

def runIdentification(Wt, automatic_deletion=False):
    params = ["cf", "cm", "ms_x", "ms_y", "ms_z", "Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz"]

    for i in range(10):
        Xt_hat, Cxx = solveTLS(Wt)

        std_dev = np.array([])
        for idx in range(Cxx.shape[0]):
            std_dev = np.append(std_dev, 100 * np.sqrt(Cxx[idx, idx]) / np.abs(Xt_hat[idx]))

        printResults(params, Xt_hat, std_dev)
        if automatic_deletion:
            del_idx = np.argmax(std_dev)
        else:
            name = input("Enter the name of the parameter to delete: ")
            if name == "end":
                return
            del_idx = params.index(name)

        param_deleted = params.pop(del_idx)
        Wt = np.delete(Wt, del_idx, axis=1)
        print("\nIteration", i, "param to be deleted:", param_deleted)


def solveTLS(Wt):
    u, s, vh = np.linalg.svd(Wt, full_matrices=True)
    Xt_hat_star = vh[:, -1]
    Xt_hat = Xt_hat_star / Xt_hat_star[-1]

    nt = Wt.shape[1]
    r = Wt.shape[0]
    s_nt = min(s)
    sigma_hat_w = s_nt / np.sqrt(r - nt)

    Wt_bar = Wt - s_nt * np.array([u[:, -1]]).T @ np.array([vh[:, -1]])
    Wt_bar_inv = np.linalg.inv(Wt_bar[:, :-1].T @ Wt_bar[:, :-1])
    
    Cxx = sigma_hat_w**2 * (1 + np.linalg.norm(Xt_hat[:-1])**2) * Wt_bar_inv
    
    return Xt_hat, Cxx
