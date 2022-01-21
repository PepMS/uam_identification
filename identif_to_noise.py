import numpy as np
import matplotlib.pyplot as plt

import pinocchio
import crocoddyl
import eagle_mpc

from eagle_mpc.utils.robots_loader import load

import identification

#######################
# Trajectory Creation #
#######################

dt = 20  # ms
robot_name = 'iris'
trajectory_name = 'displacement_identif'

trajectory = eagle_mpc.Trajectory()
trajectory.autoSetup(robot_name + "/trajectories/" + trajectory_name + ".yaml")
problem = trajectory.createProblem(dt, True, "IntegratedActionModelEuler")

solver = eagle_mpc.SolverSbFDDP(problem, trajectory.squash)

solver.setCallbacks([crocoddyl.CallbackVerbose()])
solver.solve([], [], maxiter=100)

xs = solver.xs
us = solver.us_squash
ns = [np.sqrt(u / trajectory.platform_params.cf) for u in us]
accs = [d.differential.xout for d in solver.problem.runningDatas]

#############
# Add Noise #
#############

qs_n = []
vs_n = []
ws_n = []
accs_n = []
ns_n = []

q_std = 1e-2
v_std = 1e-2
w_std = 5 * np.pi / 180
al_std = 5e-2
aw_std = 5 * np.pi / 180
n_std = 5

# q_std = 0
# v_std = 0
# w_std = 0
# al_std = 0
# aw_std = 0
# n_std = 0

for idx, (x, n, a) in enumerate(zip(xs, ns, accs)):

    dq = np.random.normal(0, q_std, 3)

    qs_n.append(pinocchio.Quaternion(x[3:7]) * pinocchio.Quaternion(pinocchio.exp3(dq)))
    vs_n.append(x[7:10] + np.random.normal(0, v_std, 3))  # st. dev 5 cm/s
    ws_n.append(x[10:13] + np.random.normal(0, w_std, 3))  # st. dev 5 deg/s

    acc = np.copy(a)
    acc[:3] = acc[:3] + np.random.normal(0, al_std, 3)  # st. dev 0.5 m/s^2
    acc[3:] = acc[3:] + np.random.normal(0, aw_std, 3)  # st. dev 1 deg/s^2
    accs_n.append(acc)

    ns_n.append(n + np.random.normal(0, n_std, trajectory.platform_params.n_rotors))

# # Quaternion
# fig0, axs0 = plt.subplots(2)
# axs0[0].plot([x[3:7] for x in xs])
# axs0[1].plot([q.coeffs() for q in qs_n])

# # Linear Velocity
# fig1, axs1 = plt.subplots(2)
# axs1[0].plot([x[7:10] for x in xs])
# axs1[1].plot(vs_n)

# # Angular Velocity
# fig2, axs2 = plt.subplots(2)
# axs2[0].plot([x[10:13] for x in xs])
# axs2[1].plot(ws_n)

# # Linear Acc
# fig3, axs3 = plt.subplots(2)
# axs3[0].plot([a[:3] for a in accs])
# axs3[1].plot([a[:3] for a in accs_n])

# # Angular Acc
# fig4, axs4 = plt.subplots(2)
# axs4[0].plot([a[3:] for a in accs])
# axs4[1].plot([a[3:] for a in accs_n])

# # Rotors
# fig5, axs5 = plt.subplots(2)
# axs5[0].plot(ns)
# axs5[1].plot(ns_n)

# plt.show()

##################
# Identification #
##################

W_lst = []
Wm_lst = []
Wk_lst = []

for idx, (q, v, w, a, n) in enumerate(zip(qs_n, vs_n, ws_n, accs_n, ns_n)):

    a_lin = a[:3] + identification.skew(w) @ v

    D = identification.computeD(q.toRotationMatrix(), w, a_lin, a[3:])
    Dm = identification.computeDm(q.toRotationMatrix(), a_lin)
    Dk = identification.computeDk(n)

    W_lst.append(D)
    Wm_lst.append(Dm)
    Wk_lst.append(Dk)

m = 1.52
Wt = np.concatenate([np.vstack(Wk_lst), np.vstack(W_lst), m * np.vstack(Wm_lst)], axis=1)

identification.runIdentification(Wt)

print("\nReal parameters: ")
dyn_param = trajectory.robot_model.inertias[1].toDynamicParameters()

params = ["cf", "cm", "ms_x", "ms_y", "ms_z", "Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz"]
values = np.array([
    trajectory.platform_params.cf, trajectory.platform_params.cm, dyn_param[1], dyn_param[2], dyn_param[3], dyn_param[4], dyn_param[6], dyn_param[9],
    dyn_param[5], dyn_param[7], dyn_param[8]
])

for (param, value) in zip(params, values):
    print(param, ":", value)