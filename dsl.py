import numpy as np
from numpy.lib import type_check
import pinocchio
import crocoddyl
import eagle_mpc
from eagle_mpc.utils.robots_loader import load
import identification
m_net = []
m_net.append(np.array([0.030443, 0.020971, 0.006571]))
m_net.append(np.array([0.015789, 0.002978, -0.006981]))
m_net.append(np.array([-0.005204, -0.009218, -0.055764]))
m_net.append(np.array([0.048196, 0.030422, 0.049688]))

m_thrust = []
m_thrust.append(1.10472)
m_thrust.append(4.36937)
m_thrust.append(0.110925)
m_thrust.append(10.3401)

m_grav = 0.005 * np.array([0, 0, -9.81])

q = pinocchio.Quaternion(np.array([-0.0924809, 0.176149, -0.0446938, 0.97899]))

m_reaction = []
for (m_n, th) in zip(m_net, m_thrust):
    # Store reaction force in the platform, thus multiply *-1
    m_reaction.append(-(m_n - q.toRotationMatrix() @ np.array([0, 0, th]) - m_grav))
    print(m_reaction[-1])

p_net = np.zeros(3)
for m_r in m_reaction:
    p_net += m_r

p_net += 1.5 * np.array([0, 0, -9.81])

print("Net platform: ", p_net)

robot = load("iris")
r_model = robot.model
r_data = robot.data

mc_params = eagle_mpc.MultiCopterBaseParams()
mc_params.autoSetup("/home/pepms/robotics/libraries/eagle-mpc/yaml/iris/platform/iris.yaml", r_model)

r_state = crocoddyl.StateMultibody(r_model)

act_model = crocoddyl.ActuationModelMultiCopterBase(r_state, mc_params.tau_f)
act_data = act_model.createData()

state = np.zeros(13)
state[:3] = np.array([0.763331, 0.827915, 1.0189])
q = pinocchio.Quaternion(np.array([-0.0707727, 0.228294, -0.0716893, 0.968367]))
q.normalize()
state[3:7] = q.coeffs()
state[7:10] = np.array([1.35111, 3.2002, 4.76434])
state[10:13] = np.array([2.64566, -1.1205, -0.034687])

thrust = np.array([6.19517, 4.07863, 10.3306, 9.75763])
act_model.calc(act_data, state, thrust)

tau = act_data.tau

a_model = pinocchio.aba(r_model, r_data, state[:r_model.nq], state[r_model.nq:], tau)

net_force = tau[:3] + r_model.inertias[1].mass * (q.toRotationMatrix().T @ np.array([0, 0, -9.81]))
net_torque = tau[3:] + r_model.inertias[1].mass * identification.skew(
    r_model.inertias[1].lever) @ (q.toRotationMatrix().T @ np.array([0, 0, -9.81]))

net_torque = q.toRotationMatrix().T @ np.array([-0.313381, -0.390004, 0.362522])

a_ang_pin = np.linalg.inv(r_model.inertias[1].inertia) @ (
    net_torque - identification.skew(state[10:13]) @ r_model.inertias[1].inertia @ state[10:13])

print(a_model[:3] + identification.skew(state[10:13]) @ state[7:10])
print(a_model[3:])