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

for i in range(5000):

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

    D, Dm = identification.computeDDm(R, v_motion, a_motion)
    Dk = identification.computeDk(n_rotors)

    W_lst.append(D)
    Wm_lst.append(np.array([Dm]).T)
    Wk_lst.append(-Dk)

m = 1.52
Wt = np.concatenate([np.vstack(Wk_lst), np.vstack(W_lst), m * np.vstack(Wm_lst)], axis=1)

identification.runIdentification(Wt)