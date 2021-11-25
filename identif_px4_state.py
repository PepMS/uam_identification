import numpy as np
import pinocchio
import matplotlib.pyplot as plt

import identification

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
file_name = '/home/pepms/robotics/method-test/uam_identification/csvs/3_minutes.csv'

prop_min_speed = 100  # Only valid for this simulation
prop_max_speed = 1100  # Only valid for this simulation

data = identification.multicopterData(file_name, prop_min_speed, prop_max_speed)

rotor_rads, rotor_ts = data.loadRotorAngularVelocity()
states, state_ts = data.loadPlatformState()
a_lin_flus, a_lin_ts = data.loadAcceleration()
a_ang_flus, a_ang_ts = data.loadAngularAcceleration()

print("Length of rotors: ", len(rotor_rads))
print("Length of state: ", len(states))
print("Length of linear acceleration: ", len(a_lin_flus))
print("Length of angular acceleration: ", len(a_ang_flus))

print('\nCreating matrices...\n')

idx_min = 0
idx_max = -1

W_lst = []
Wm_lst = []
Wk_lst = []

errors = []
a_lin_nogs = []
t_ini = min(state_ts[0], rotor_ts[0], a_lin_ts[0], a_ang_ts[0])
state_ts = [x - t_ini for x in state_ts]
rotor_ts = [x - t_ini for x in rotor_ts]
a_lin_ts = [x - t_ini for x in a_lin_ts]
a_ang_ts = [x - t_ini for x in a_ang_ts]

for idx, (rotor, state, a_lin, a_ang) in enumerate(
        zip(rotor_rads[idx_min:idx_max], states[idx_min:idx_max], a_lin_flus[idx_min:idx_max],
            a_ang_flus[idx_min:idx_max])):

    q = pinocchio.Quaternion(state[3:7])

    a = a_lin + q.toRotationMatrix().T @ np.array([0, 0, -9.81])
    a_lin_nogs.append(a)

    v_motion = pinocchio.Motion(state[7:10], state[10:13])
    a_motion = pinocchio.Motion(a, a_ang)

    D, Dm = identification.computeDDm(q.toRotationMatrix(), v_motion, a_motion)
    Dk = identification.computeDk(rotor)

    errors.append(D @ X + m * Dm - Dk @ Xk)

    W_lst.append(D)
    Wm_lst.append(np.array([Dm]).T)
    Wk_lst.append(-Dk)

# plt.figure()
# plt.plot(rotor_ts[idx_min:idx_max], rotor_rads[idx_min:idx_max])
# plt.figure()
# plt.plot(state_ts[idx_min:idx_max], [state[3:7] for state in states[idx_min:idx_max]])
# plt.figure()
# plt.plot(state_ts[idx_min:idx_max], [state[7:10] for state in states[idx_min:idx_max]])
# plt.figure()
# plt.plot(state_ts[idx_min:idx_max], [state[10:13] for state in states[idx_min:idx_max]])
# plt.figure()
# plt.plot(a_lin_ts[idx_min:idx_max], a_lin_nogs)
# plt.figure()
# plt.plot(a_ang_ts[idx_min:idx_max], a_ang_flus[idx_min:idx_max])

# plt.figure()
# plt.plot([np.linalg.norm(error) for error in errors])

plt.show()

m = 1.52
Wt = np.concatenate([np.vstack(Wk_lst), np.vstack(W_lst), m * np.vstack(Wm_lst)], axis=1)

print('\nIdentifying...\n')
identification.runIdentification(Wt)

print('\nReal parameters...\n')
dyn_param = r_model.inertias[1].toDynamicParameters()

params = ["cf", "cm", "ms_x", "ms_y", "ms_z", "Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz"]
values = np.array([5.84e-06, 3.504e-7, dyn_param[1], dyn_param[2], dyn_param[3], dyn_param[4], dyn_param[6], dyn_param[9], dyn_param[5], dyn_param[7],
    dyn_param[8]
])

for (param, value) in zip(params, values):
    print(param, ":", value)

