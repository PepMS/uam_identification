import numpy as np
import eigenpy
import matplotlib.pyplot as plt

import identification

# Quadcopter description:
#        y
#    2   ^   3
#     \  |  /
#      \ | /
#       \|/
#        ------> x
#       / \
#      /   \
#     /     \
#    4       1
#
# Rotor Coordinates (x, y) w.r.t. the IMU location
# Rotor 1: (0.13, -0.22), CCW (Counter Clock Wise)
# Rotor 2: (-0.13, 0.2), CCW
# Rotor 3: (0.13, 0.22), CW (Clock Wise)
# Rotor 4: (-0.13, -0.2)
#
# CCW (CW) results in a negative (positive) reactive torque along the Z direction. See equation 9 in the paper.
#
# Mass: 1.5 kg

print('Data loading...')
file_name = './csvs/identification.csv'

prop_min_speed = 100  # Only valid for this simulation
prop_max_speed = 1100  # Only valid for this simulation

data = identification.multicopterData(file_name, prop_min_speed, prop_max_speed)

rotor_rads, rotor_ts = data.loadRotorAngularVelocity()
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

idx_max = 5000

W_lst = []
Wm_lst = []
Wk_lst = []

# for idx, (rotor, acc, ang_vel, ang_acc, rpy) in enumerate(
#         zip(rotor_rads, acc_flus[1:idx_max], ang_vel_flus[1:idx_max], ang_acc_flus[1:idx_max], rpys[2:idx_max])):
#     W_lst.append(identification.computeD(acc, rpy, ang_vel, ang_acc))
#     Wk_lst.append(identification.computeDk(rotor))
#     Wm_lst.append(np.array([identification.computeDm(acc, rpy)]).T)
#     if idx == 3400:
#       a = 0

for idx, (rotor, acc, ang_vel, ang_acc, R) in enumerate(
        zip(rotor_rads, acc_flus[1:idx_max], ang_vel_flus[1:idx_max], ang_acc_flus[1:idx_max], R_nwu_flus[2:idx_max])):
    W_lst.append(identification.computeD(acc, R, ang_vel, ang_acc))
    Wk_lst.append(-identification.computeDk(rotor))
    Wm_lst.append(np.array([identification.computeDm(acc, R)]).T)

m = 1.5
Wt = np.concatenate([np.vstack(Wk_lst), np.vstack(W_lst), m * np.vstack(Wm_lst)], axis=1)

identification.runIdentification(Wt)

