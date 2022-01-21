import numpy as np

import pinocchio
import crocoddyl

import eagle_mpc
from eagle_mpc.utils.robots_loader import load

import identification

# Library objects
robot = load("iris")
r_model = robot.model
r_data = robot.data

mc_params = eagle_mpc.MultiCopterBaseParams()
mc_params.autoSetup("/home/pepms/robotics/libraries/eagle-mpc/yaml/iris_px4/platform/iris_px4.yaml", r_model)

r_state = crocoddyl.StateMultibody(r_model)

act_model = crocoddyl.ActuationModelMultiCopterBase(r_state, mc_params.tau_f)
act_data = act_model.createData()

# Data Creation
x = r_state.rand()
x[:3] = np.random.rand(3)

u = mc_params.u_lb + np.random.rand(mc_params.n_rotors) * (mc_params.u_ub - mc_params.u_lb)
n_rotors = np.sqrt(u / mc_params.cf)
act_model.calc(act_data, x, u)
tau = act_data.tau

p = x[:3]
q = pinocchio.Quaternion(x[6], x[3], x[4], x[5])
R = q.toRotationMatrix()
v = x[7:10]
w = x[10:]

# Parameters
Ic = np.array([[3.56546408e-02, -2.71050543e-20, 0.00000000e+00], [-2.71050543e-20, 4.73337568e-02, -3.38813179e-21],
               [0.00000000e+00, -3.38813179e-21, 1.00018016e-01]])
m = 1.5199999999999996
c = np.array([0., 0., 0.00030263])

# Simulating
a_3d = identification.fd3dVectors(q.toRotationMatrix(), v, w, tau[:3], tau[3:], Ic, c, m)

a_lin = a_3d[:3]
a_ang = a_3d[3:]

##############################
# Regressor loading - Custom #
##############################

Dk = identification.computeDk(n_rotors)
D = identification.computeD(R, w, a_lin, a_ang)
Dm = identification.computeDm(R, a_lin)

# Parameter Vector
Ib = Ic + m * identification.skew(c) @ identification.skew(c).T

X = np.array([m * c[0], m * c[1], m * c[2], Ib[0, 0], Ib[1, 1], Ib[2, 2], Ib[0, 1], Ib[0, 2], Ib[1, 2]])
Xk = np.array([mc_params.cf, mc_params.cm])

D_res = D @ X + Dk @ Xk + m * Dm.T
print("Regressor residual: ", D_res)
print("Regressor residual norm: ", np.linalg.norm(D_res))

#################################
# Regressor loading - Pinocchio #
#################################
a_lin = a_3d[:3] - identification.skew(w) @ v
a_ang = a_3d[3:] 

v_motion = pinocchio.Motion(v, w)
a_motion = pinocchio.Motion(a_lin, a_ang)

D, Dm = identification.computeDDm(R, v_motion, a_motion)
Dk = identification.computeDk(n_rotors)

X = np.array([m * c[0], m * c[1], m * c[2], Ic[0, 0], Ic[1, 1], Ic[2, 2], Ic[0, 1], Ic[0, 2], Ic[1, 2]])
Xk = np.array([mc_params.cf, mc_params.cm])

D_res = D @ X - Dk @ Xk + m * Dm.T
print("Regressor residual: ", D_res)
print("Regressor residual norm: ", np.linalg.norm(D_res))