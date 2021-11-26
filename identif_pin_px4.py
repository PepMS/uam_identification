import numpy as np
import matplotlib.pyplot as plt

import identification
import pinocchio
import crocoddyl
import eagle_mpc

from eagle_mpc.utils.robots_loader import load

print('Data loading...')
file_name = '/home/pepms/robotics/method-test/uam_identification/csvs/3_minutes.csv'

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
idx_max = 2000

as_model = []
a_lin_nogs = []

for idx, (rotor, state, a_lin, a_ang) in enumerate(
        zip(rotor_rads[idx_min:idx_max], states[idx_min:idx_max], a_lin_flus[idx_min:idx_max],
            a_ang_flus[idx_min:idx_max])):

    q = pinocchio.Quaternion(state[3:7])

    a = a_lin + q.toRotationMatrix().T @ np.array([0, 0, -9.81])
    a_lin_nogs.append(a)

    v_motion = pinocchio.Motion(state[7:10], state[10:13])
    a_motion = pinocchio.Motion(a, a_ang)

    thrust = rotor**2 * mc_params.cf
    act_model.calc(act_data, state, thrust)
    tau = act_data.tau

    a_model = pinocchio.aba(r_model, r_data, state[:r_model.nq], state[r_model.nq:], tau)
    as_model.append(a_model[:3])

fig, axs = plt.subplots(3)
axs[0].plot([a[0] for a in a_lin_nogs[idx_min:idx_max]])
axs[0].plot([a[0] for a in as_model[idx_min:idx_max]])

axs[1].plot([a[1] for a in a_lin_nogs[idx_min:idx_max]])
axs[1].plot([a[1] for a in as_model[idx_min:idx_max]])

axs[2].plot([a[2] for a in a_lin_nogs[idx_min:idx_max]])
axs[2].plot([a[2] for a in as_model[idx_min:idx_max]])
plt.show()   

