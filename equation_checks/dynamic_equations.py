import numpy as np

import pinocchio
import identification

from eagle_mpc.utils.robots_loader import load


# Function definition
def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def spatial_skew(v):
    r1 = np.hstack([skew(v[3:]), skew(v[:3])])
    r2 = np.hstack([np.zeros([3, 3]), skew(v[3:])])

    return np.vstack([r1, r2])


def spatial_skew_conj(v):
    return -spatial_skew(v).T


# FD implementation

# Data
q = pinocchio.Quaternion(np.array([-0.0707727, 0.228294, -0.0716893, 0.968367]))
q.normalize()
v = np.array([1.35111, 3.2002, 4.76434])
w = np.array([2.64566, -1.1205, -0.034687])

tau = np.array([0., 0., 19.68595326, 0.56437819, -0.73561917, -0.15584124])

# Parameters
Ic = np.array([[3.56546408e-02, -2.71050543e-20, 0.00000000e+00], [-2.71050543e-20, 4.73337568e-02, -3.38813179e-21],
               [0.00000000e+00, -3.38813179e-21, 1.00018016e-01]])
m = 1.5199999999999996
c = np.array([0., 0., 0.00030263])

#######################
# Manual Featherstone #
#######################
# Spatial vectors [linear, angular]

# Spatial inertia
Ir1 = np.hstack([m * np.identity(3), m * skew(c).T])
Ir2 = np.hstack([m * skew(c), Ic + m * skew(c) @ skew(c).T])
# Ir2 = np.hstack([m * skew(c), Ic])
Isp = np.vstack([Ir1, Ir2])

v_sp = np.hstack([v, w])
g_sp = np.hstack([q.toRotationMatrix().T @ np.array([0, 0, -9.81]), np.zeros(3)])
a_sp = np.linalg.inv(Isp) @ (Isp @ g_sp + tau - spatial_skew_conj(v_sp) @ Isp @ v_sp)

# f_res = np.hstack([tau[:3] + m*q.toRotationMatrix().T @ np.array([0, 0, -9.81]), tau[3:]])
# a_sp = np.linalg.inv(Isp) @ (f_res - spatial_skew_conj(v_sp) @ Isp @ v_sp)

print("Spatial acceleration: \n", a_sp)

#############
# Pinocchio #
#############

robot = load("iris")
r_model = robot.model
r_data = robot.data

a_pin = pinocchio.aba(r_model, r_data, np.hstack([np.zeros(3), q.coeffs()]), v_sp, tau)

print("Pinocchio acceleration: \n", a_pin)


##############
# 3D vectors #
##############
a_3d = identification.fd3dVectors(q.toRotationMatrix(),v,w,tau[:3], tau[3:], Ic, c, m)

a_3d_sp = a_3d
a_3d_sp[:3] = a_3d[:3] - skew(w) @ v


print("3D Vector acceleration: \n", a_3d_sp)

print("Spatial-Pinocchio difference: ", np.linalg.norm(a_sp - a_pin))
print("3D Vector-Pinocchio difference: ", np.linalg.norm(a_3d_sp - a_pin))
print("3D Vector-Spatial difference: ", np.linalg.norm(a_3d_sp - a_sp))