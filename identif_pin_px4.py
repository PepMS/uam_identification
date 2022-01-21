import numpy as np
import matplotlib.pyplot as plt

import identification
import pinocchio
import crocoddyl
import eagle_mpc

from eagle_mpc.utils.robots_loader import load

print('Data loading...')
file_name = './csvs/long.csv'

prop_min_speed = 100  # Only valid for this simulation
prop_max_speed = 1100  # Only valid for this simulation

data = identification.multicopterData(file_name, prop_min_speed, prop_max_speed)

rotor_rads, rotor_ts = data.loadRotorAngularVelocity()
states, state_ts = data.loadPlatformState()
a_lin_flus, a_lin_ts = data.loadAcceleration()
a_ang_flus, a_ang_ts = data.loadAngularAcceleration()

print('Robot loading...')

robot = load("iris_px4")
r_model = robot.model
r_data = robot.data

mc_params = eagle_mpc.MultiCopterBaseParams()
mc_params.autoSetup("/home/pepms/robotics/libraries/eagle-mpc/yaml/iris_px4/platform/iris_px4.yaml", r_model)

r_state = crocoddyl.StateMultibody(r_model)

act_model = crocoddyl.ActuationModelMultiCopterBase(r_state, mc_params.tau_f)
act_data = act_model.createData()

W_lst = []
Wm_lst = []
Wk_lst = []

idx_min = 0
idx_max = -1

as_model = []
a_lin_nogs = []

for idx, (rotor, state, a_lin, a_ang) in enumerate(
        zip(rotor_rads[idx_min:idx_max], states[idx_min:idx_max], a_lin_flus[idx_min:idx_max],
            a_ang_flus[idx_min:idx_max])):

    q = pinocchio.Quaternion(state[3:7])

    a = a_lin + q.toRotationMatrix().T @ np.array([0, 0, -9.81])
    a_lin_nogs.append(a)

    thrust = rotor**2 * mc_params.cf
    act_model.calc(act_data, state, thrust)
    tau = act_data.tau

    a_model = pinocchio.aba(r_model, r_data, state[:r_model.nq], state[r_model.nq:], tau)
    a_model[:3] = a_model[:3] + identification.skew(state[10:13]) @ state[7:10]
    as_model.append(a_model)

    D = identification.computeD(q.toRotationMatrix(), state[10:13], a, a_ang)
    Dm = identification.computeDm(q.toRotationMatrix(), a)
    Dk = identification.computeDk(rotor)

    W_lst.append(D)
    Wm_lst.append(Dm)
    Wk_lst.append(Dk)

fig0, axs0 = plt.subplots(4)
axs0[0].plot([a[0] for a in a_lin_nogs[idx_min:idx_max]])
axs0[0].plot([a[0] for a in as_model[idx_min:idx_max]])

axs0[1].plot([a[1] for a in a_lin_nogs[idx_min:idx_max]])
axs0[1].plot([a[1] for a in as_model[idx_min:idx_max]])

axs0[2].plot([a[2] for a in a_lin_nogs[idx_min:idx_max]])
axs0[2].plot([a[2] for a in as_model[idx_min:idx_max]])

axs0[3].plot([np.linalg.norm(a) for a in a_lin_nogs[idx_min:idx_max]])
axs0[3].plot([np.linalg.norm(a) for a in as_model[idx_min:idx_max]])

axs0[0].legend(['px4_acc_lin', 'pinocchio_acc_lin'])

fig1, axs1 = plt.subplots(3)
axs1[0].plot([a[0] for a in a_ang_flus[idx_min:idx_max]])
axs1[0].plot([a[3] for a in as_model[idx_min:idx_max]])

axs1[1].plot([a[1] for a in a_ang_flus[idx_min:idx_max]])
axs1[1].plot([a[4] for a in as_model[idx_min:idx_max]])

axs1[2].plot([a[2] for a in a_ang_flus[idx_min:idx_max]])
axs1[2].plot([a[5] for a in as_model[idx_min:idx_max]])

axs1[0].legend(['px4_acc_ang', 'pinocchio_acc_ang'])

plt.show()

m = 1.52
Wt = np.concatenate([np.vstack(Wk_lst), np.vstack(W_lst), m * np.vstack(Wm_lst)], axis=1)

print('\nIdentifying...\n')
identification.runIdentification(Wt)

print('\nReal parameters...\n')
dyn_param = r_model.inertias[1].toDynamicParameters()

params = ["cf", "cm", "ms_x", "ms_y", "ms_z", "Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz"]
values = np.array([
    5.84e-06, 3.504e-7, dyn_param[1], dyn_param[2], dyn_param[3], dyn_param[4], dyn_param[6], dyn_param[9],
    dyn_param[5], dyn_param[7], dyn_param[8]
])

for (param, value) in zip(params, values):
    print(param, ":", value)