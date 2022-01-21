import csv
from os import error
import numpy as np
import pinocchio

import fbpca


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def fd3dVectors(R, v, w, tau_lin, tau_ang, Ic, c, m):
    g_w = np.array([0, 0, -9.81])
    f_res1 = tau_lin + m * R.T @ g_w + m * skew(w) @ skew(c) @ w
    f_res2 = tau_ang + m * skew(c) @ R.T @ g_w - skew(w) @ (Ic + m * skew(c) @ skew(c).T) @ w

    f_res = np.hstack([f_res1, f_res2])

    Ir1 = np.hstack([m * np.identity(3), m * skew(c).T])
    Ir2 = np.hstack([m * skew(c), Ic + m * skew(c) @ skew(c).T])
    Isp = np.vstack([Ir1, Ir2])

    a_3d = np.linalg.inv(Isp) @ f_res

    return a_3d


class multicopterData():
    def __init__(self, file_name, prop_min_speed=100, prop_max_speed=1100, ros2=True):
        self.file_name = file_name
        self.time_name = 'timestamp'

        self.prop_min_speed = prop_min_speed
        self.prop_max_speed = prop_max_speed
        self.ros2 = ros2

    def loadNumericalData(self, name_list, name_timestamp='__time'):
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
                        values_ts.append(float(row[name_timestamp]))
        return values, values_ts

    def loadPlatformState(self):
        if self.ros2:
            state_names = [
                '/platform_state/pose/position/x', '/platform_state/pose/position/y',
                '/platform_state/pose/position/z', '/platform_state/pose/orientation/x',
                '/platform_state/pose/orientation/y', '/platform_state/pose/orientation/z',
                '/platform_state/pose/orientation/w', '/platform_state/motion/linear/x',
                '/platform_state/motion/linear/y', '/platform_state/motion/linear/z',
                '/platform_state/motion/angular/x', '/platform_state/motion/angular/y',
                '/platform_state/motion/angular/z'
            ]
            time_name = '__time'
        else:
            state_names = [
                '/iris/odometry_sensor1/odometry/pose/pose/position/x',
                '/iris/odometry_sensor1/odometry/pose/pose/position/y',
                '/iris/odometry_sensor1/odometry/pose/pose/position/z',
                '/iris/odometry_sensor1/odometry/pose/pose/orientation/x',
                '/iris/odometry_sensor1/odometry/pose/pose/orientation/y',
                '/iris/odometry_sensor1/odometry/pose/pose/orientation/z',
                '/iris/odometry_sensor1/odometry/pose/pose/orientation/w',
                '/iris/odometry_sensor1/odometry/twist/twist/linear/x',
                '/iris/odometry_sensor1/odometry/twist/twist/linear/y',
                '/iris/odometry_sensor1/odometry/twist/twist/linear/z',
                '/iris/odometry_sensor1/odometry/twist/twist/angular/x',
                '/iris/odometry_sensor1/odometry/twist/twist/angular/y',
                '/iris/odometry_sensor1/odometry/twist/twist/angular/z'
            ]
            time_name = '/iris/odometry_sensor1/odometry/header/stamp'
        states, state_ts = self.loadNumericalData(state_names, time_name)

        return states, state_ts

    def loadLinearVelocity(self):
        # Angular acceleration
        rotor_names = [
            '/fmu/vehicle_local_position_groundtruth/out/vx', '/fmu/vehicle_local_position_groundtruth/out/vy',
            '/fmu/vehicle_local_position_groundtruth/out/vz'
        ]
        vels, vel_ts = self.loadNumericalData(rotor_names)

        return [fromFRDtoFLU(vel) for vel in vels], vel_ts

    def loadRotorAngularVelocity(self):
        # Angular acceleration
        if self.ros2:
            rotor_names = [
                '/fmu/actuator_outputs/out/output.0',
                '/fmu/actuator_outputs/out/output.1',
                '/fmu/actuator_outputs/out/output.2',
                '/fmu/actuator_outputs/out/output.3',
            ]
        else:
            rotor_names = [
                '/iris/motor_speed/angular_velocities.0',
                '/iris/motor_speed/angular_velocities.1',
                '/iris/motor_speed/angular_velocities.2',
                '/iris/motor_speed/angular_velocities.3',
            ]
        rotor_pwms, rotor_ts = self.loadNumericalData(rotor_names)

        if self.ros2:
            return [fromPWMtoRadS(rotor_pwm, self.prop_min_speed, self.prop_max_speed)
                    for rotor_pwm in rotor_pwms], rotor_ts
        else:
            return rotor_pwms, rotor_ts

    def loadAcceleration(self):
        # Accelerations
        if self.ros2:
            acc_names = [
                '/fmu/sensor_combined/out/accelerometer_m_s2.0', '/fmu/sensor_combined/out/accelerometer_m_s2.1',
                '/fmu/sensor_combined/out/accelerometer_m_s2.2'
            ]
        else:
            acc_names = [
                '/iris/imu/linear_acceleration/x', '/iris/imu/linear_acceleration/y', '/iris/imu/linear_acceleration/z'
            ]
        acc_frds, acc_ts = self.loadNumericalData(acc_names)

        return [fromFRDtoFLU(acc) for acc in acc_frds], acc_ts

    def loadAngularVelocity(self):
        # Angular velocity
        ang_vel_names = [
            '/fmu/sensor_combined/out/gyro_rad.0', '/fmu/sensor_combined/out/gyro_rad.1',
            '/fmu/sensor_combined/out/gyro_rad.2'
        ]
        ang_vel_frds, ang_vel_ts = self.loadNumericalData(ang_vel_names)

        return [fromFRDtoFLU(ang_vel) for ang_vel in ang_vel_frds], ang_vel_ts

    def loadAngularAcceleration(self):
        # Angular velocity
        ang_acc_names = [
            '/fmu/vehicle_angular_acceleration/out/xyz.0', '/fmu/vehicle_angular_acceleration/out/xyz.1',
            '/fmu/vehicle_angular_acceleration/out/xyz.2'
        ]
        ang_acc_frds, ang_acc_ts = self.loadNumericalData(ang_acc_names)

        return [fromFRDtoFLU(ang_acc) for ang_acc in ang_acc_frds], ang_acc_ts

    def loadAttitude(self):
        quat_names = [
            '/fmu/vehicle_attitude_groundtruth/out/q.0', '/fmu/vehicle_attitude_groundtruth/out/q.1',
            '/fmu/vehicle_attitude_groundtruth/out/q.2', '/fmu/vehicle_attitude_groundtruth/out/q.3'
        ]

        quat_ned_frds, quat_ts = self.loadNumericalData(quat_names)

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


def computeD(R, w, a_lin, a_ang):
    g_w = R.T @ np.array([0, 0, -9.81])

    com_1 = skew(a_ang).T + skew(w) @ skew(w).T
    com_2 = skew(g_w).T - skew(a_lin).T
    com = np.vstack([com_1, com_2])

    # Inertia
    i_1 = np.zeros([3, 6])

    i_21_1 = np.diag(a_ang)
    i_22_1 = np.array([[a_ang[1], a_ang[2], 0], [a_ang[0], 0, a_ang[2]], [0, a_ang[0], a_ang[1]]])
    i_2_1 = np.hstack([i_21_1, i_22_1])

    i_21_2 = np.diag(w)
    i_22_2 = np.array([[w[1], w[2], 0], [w[0], 0, w[2]], [0, w[0], w[1]]])
    i_2_2 = np.hstack([i_21_2, i_22_2])

    i_2 = -i_2_1 - skew(w) @ i_2_2

    i = np.vstack([i_1, i_2])

    return np.hstack([com, i])


def computeDm(R, a_lin):
    g_w = R.T @ np.array([0, 0, -9.81])

    m_1 = g_w - a_lin

    return np.vstack([np.array([m_1]).T, np.zeros([3, 1])])



def computeDDm(R, vel, acc):
    regressor = pinocchio.bodyRegressor(vel, acc)

    Dm = np.zeros(6)
    Dm[:3] = regressor[:3, 0] - R.T @ np.array([0, 0, -9.81])
    Dm[3:] = regressor[3:, 0]

    D = np.vstack([
        regressor[:, 1], regressor[:, 2], regressor[:, 3], regressor[:, 4], regressor[:, 6], regressor[:, 9],
        regressor[:, 5], regressor[:, 7], regressor[:, 8]
    ]).T

    # Displaced center of gravity generates moment
    D[3:, :3] = D[3:, :3] - skew(-R.T @ np.array([0, 0, -9.81]))

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

    for i in range(11):
        Xt_hat, Cxx = solveTLS(Wt)

        std_dev = np.array([])
        for idx in range(Cxx.shape[0]):
            # std_dev = np.append(std_dev, 100 * np.sqrt(Cxx[idx, idx]) / np.abs(Xt_hat[idx]))
            std_dev = np.append(std_dev, np.sqrt(Cxx[idx, idx]))

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
    u, s, vh = fbpca.pca(Wt, k=Wt.shape[1], raw=False, n_iter=2, l=None)
    # u, s, vh = np.linalg.svd(Wt, full_matrices=True)
    # Xt_hat_star = vh[:, -1]
    Xt_hat_star = vh[-1, :]
    Xt_hat = Xt_hat_star / Xt_hat_star[-1]

    nt = Wt.shape[1]
    r = Wt.shape[0]
    s_nt = min(s)
    sigma_hat_w = s_nt / np.sqrt(r - nt)

    Wt_bar = Wt - s_nt * np.array([u[:, -1]]).T @ np.array([vh[-1, :]])
    Wt_bar_inv = np.linalg.inv(Wt_bar[:, :-1].T @ Wt_bar[:, :-1])

    Cxx = sigma_hat_w**2 * (1 + np.linalg.norm(Xt_hat[:-1])**2) * Wt_bar_inv

    return Xt_hat, Cxx
