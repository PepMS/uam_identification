import numpy as np
import eigenpy
import pinocchio
import matplotlib.pyplot as plt

import identification

print('Data loading...')
file_name = './csvs/identification.csv'

prop_min_speed = 100  # Only valid for this simulation
prop_max_speed = 1100  # Only valid for this simulation

data = identification.multicopterData(file_name, prop_min_speed, prop_max_speed)

rotor_rads, rotor_ts = data.loadRotorAngularVelocity()
vel_flus, vel_ts = data.loadLinearVelocity()
acc_flus, acc_ts = data.loadAcceleration()
ang_vel_flus, ang_vel_ts = data.loadAngularVelocity()
ang_acc_flus, ang_acc_ts = data.loadAngularAcceleration()

R_nwu_flus, R_ts = data.loadAttitude()
q_nwu_flus = [eigenpy.Quaternion(R) for R in R_nwu_flus]

rpys = []
for R in R_nwu_flus:
    ypr = eigenpy.toEulerAngles(R, 2, 1, 0)  # yaw, pitch , roll
    rpy = np.array([ypr[2], ypr[1], ypr[0]])
    rpys.append(rpy)

print('Creating matrices...')

idx_ini = 100
idx_max = 5000

W_lst = []
Wm_lst = []
Wk_lst = []

errors = []

from eagle_mpc.utils.robots_loader import load
robot = load("iris_px4")
r_model = robot.model

dyn_param = r_model.inertias[1].toDynamicParameters()
m = dyn_param[0]
X = np.array([
    dyn_param[1], dyn_param[2], dyn_param[3], dyn_param[4], dyn_param[6], dyn_param[9], dyn_param[5], dyn_param[7],
    dyn_param[8]
])
Xk = np.array([5.84e-06, 3.504e-7])

plt.figure()
plt.plot(vel_ts[idx_ini:idx_max:10], vel_flus[idx_ini:idx_max:10])

plt.figure()
plt.plot(rotor_ts[idx_ini:idx_max:10], rotor_rads[idx_ini:idx_max:10])

for idx, (rotor, vel, acc, ang_vel, ang_acc, R) in enumerate(
        zip(rotor_rads[1:idx_max], vel_flus[1:idx_max], acc_flus[1:idx_max], ang_vel_flus[1:idx_max], ang_acc_flus[1:idx_max],
            R_nwu_flus[2:idx_max])):
    v_motion = pinocchio.Motion(vel, ang_vel)
    a_motion = pinocchio.Motion(acc, ang_acc)

    D, Dm = identification.computeDDm(R, v_motion, a_motion)
    Dk = identification.computeDk(rotor)

    error = np.linalg.norm(D @ X + m * Dm - Dk @ Xk)
    errors.append(error)

plt.figure()
plt.plot(errors[idx_ini:])

plt.show()