import numpy as np
from numpy.lib.financial import _rbl
from numpy.lib.function_base import corrcoef

import pinocchio
import crocoddyl
import eagle_mpc

from eagle_mpc.utils.robots_loader import load

import identification


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

robot = load("iris_px4")
r_model = robot.model
r_data = robot.data

mc_params = eagle_mpc.MultiCopterBaseParams()
mc_params.autoSetup("/home/pepms/robotics/libraries/eagle-mpc/yaml/iris_px4/platform/iris_px4.yaml", r_model)

r_state = crocoddyl.StateMultibody(r_model)

act_model = crocoddyl.ActuationModelMultiCopterBase(r_state, mc_params.tau_f)
act_data = act_model.createData()

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

# Real data
dyn_param = r_model.inertias[1].toDynamicParameters()
m = dyn_param[0]
X = np.array([
    dyn_param[1], dyn_param[2], dyn_param[3], dyn_param[4], dyn_param[6], dyn_param[9], dyn_param[5], dyn_param[7],
    dyn_param[8]
])
Xk = np.array([5.84e-06, 3.504e-7])

body_reg = pinocchio.bodyRegressor(v_motion, a_motion)

reg = body_reg @ r_model.inertias[1].toDynamicParameters()

print("\Regressor: ")
print("linear reg: ", reg[:3] - m * q.toRotationMatrix().T @ np.array([0, 0, -9.81]))
print("linear tau: ", tau[:3])
print("angular reg: ", reg[3:])
print("angular tau: ", tau[3:])

Dk = identification.computeDk(n_rotors)
# D = identification.computeD(a_lin, R, w, a_ang)
# Dm = identification.computeDm(a_lin, R)
D, Dm = identification.computeDDm(q.toRotationMatrix(), v_motion, a_motion)

print("\nMass: ")
print("Pinocchio: ", body_reg[:, 0])
print("Identification: ", Dm)
print("Difference: ", np.linalg.norm(Dm - body_reg[:, 0]))

print("\nIxx: ")
print("Pinocchio: ", body_reg[:, 4])
print("Identification: ", D[:, 3])
print("Difference: ", np.linalg.norm(D[:, 3] - body_reg[:, 4]))

print("\nIyy: ")
print("Pinocchio: ", body_reg[:, 6])
print("Identification: ", D[:, 4])
print("Difference: ", np.linalg.norm(D[:, 4] - body_reg[:, 6]))

print("\nIzz: ")
print("Pinocchio: ", body_reg[:, 9])
print("Identification: ", D[:, 5])
print("Difference: ", np.linalg.norm(D[:, 5] - body_reg[:, 9]))

print("\nIxy: ")
print("Pinocchio: ", body_reg[:, 5])
print("Identification: ", D[:, 6])
print("Difference: ", np.linalg.norm(D[:, 6] - body_reg[:, 5]))

print("\nIxz: ")
print("Pinocchio: ", body_reg[:, 7])
print("Identification: ", D[:, 7])
print("Difference: ", np.linalg.norm(D[:, 7] - body_reg[:, 7]))

print("\nIyz: ")
print("Pinocchio: ", body_reg[:, 8])
print("Identification: ", D[:, 8])
print("Difference: ", np.linalg.norm(D[:, 8] - body_reg[:, 8]))

print("\nms_x: ")
print("Pinocchio: ", body_reg[:, 1])
print("Identification: ", D[:, 0])
print("Difference: ", np.linalg.norm(D[:, 0] - body_reg[:, 1]))

print("\nms_y: ")
print("Pinocchio: ", body_reg[:, 2])
print("Identification: ", D[:, 1])
print("Difference: ", np.linalg.norm(D[:, 1] - body_reg[:, 2]))

print("\nms_z: ")
print("Pinocchio: ", body_reg[:, 3])
print("Identification: ", D[:, 2])
print("Difference: ", np.linalg.norm(D[:, 2] - body_reg[:, 3]))

print("\nD Sum: ")
print(D @ X + Dm * m - Dk @ Xk)
print("Norm: ",np.linalg.norm(D @ X + Dm * m - Dk @ Xk))
# With random data in the state the equation is not exactly 0
# With zero state and non-zero controls this equation holds
