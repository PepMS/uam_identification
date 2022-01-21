import numpy as np

import identification
import pinocchio
import crocoddyl
import eagle_mpc

from eagle_mpc.utils.robots_loader import load

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

for i in range(10000):

    x = r_state.rand()
    x[:3] = np.random.rand(3)

    u = mc_params.u_lb + np.random.rand(mc_params.n_rotors) * (mc_params.u_ub - mc_params.u_lb)
    n_rotors = np.sqrt(u / mc_params.cf)
    act_model.calc(act_data, x, u)
    tau = act_data.tau

    accel = pinocchio.aba(r_model, r_data, x[:r_model.nq], x[r_model.nq:], tau)

    # Checking start
    p = x[:3]
    q = pinocchio.Quaternion(x[6], x[3], x[4], x[5])
    R = q.toRotationMatrix()
    v = x[7:10]
    w = x[10:]
    a_lin = accel[:3]
    a_ang = accel[3:]

    v_motion = pinocchio.Motion(v, w)
    a_motion = pinocchio.Motion(a_lin, a_ang)

    a_lin = a_lin + identification.skew(w) @ v
    D = identification.computeD(q.toRotationMatrix(), w, a_lin, a_ang)
    Dm = identification.computeDm(q.toRotationMatrix(), a_lin)
    Dk = identification.computeDk(n_rotors)

    W_lst.append(D)
    # Wm_lst.append(np.array([Dm]).T)
    Wm_lst.append(Dm)
    Wk_lst.append(Dk)

m = 1.52
Wt = np.concatenate([np.vstack(Wk_lst), np.vstack(W_lst), m * np.vstack(Wm_lst)], axis=1)

identification.runIdentification(Wt)

print("\nReal parameters: ")
dyn_param = r_model.inertias[1].toDynamicParameters()

params = ["cf", "cm", "ms_x", "ms_y", "ms_z", "Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz"]
values = np.array([
    5.84e-06, 3.504e-7, dyn_param[1], dyn_param[2], dyn_param[3], dyn_param[4], dyn_param[6], dyn_param[9],
    dyn_param[5], dyn_param[7], dyn_param[8]
])

for (param, value) in zip(params, values):
    print(param, ":", value)