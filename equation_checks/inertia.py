import numpy as np

import identification
from eagle_mpc.utils.robots_loader import load

# This file requires only one rotor at the urdf

robot = load("iris")
r_model = robot.model
r_data = robot.data

I_plat = np.diag([0.0347563, 0.0458929, 0.0977])
cog_plat = np.array([0, 0, 0])
m_plat = 1.5

I_rotor = np.diag([9.75e-07, 0.000273104, 0.000274004])
cog_rotor = np.array([0.13, -0.22, 0.023])
m_rotor = 0.005

m = m_rotor + m_plat
cog = (m_plat * cog_plat + m_rotor * cog_rotor) / m

print("Center of gravity: ", cog)
print("Difference cog: ", np.linalg.norm(cog - r_model.inertias[1].lever))

# Compute the inertia at cog
I_plat_cog = I_plat + m_plat * identification.skew(cog) @ identification.skew(cog).T
I_rotor_cog = I_rotor + m_rotor * identification.skew(cog - cog_rotor) @ identification.skew(cog - cog_rotor).T
I = I_plat_cog + I_rotor_cog

print("Inertia: ", I)
print("Inertia difference: ", np.linalg.norm(I - r_model.inertias[1].inertia))

# Pinocchio computes the inertia at the center of gravity of the (resulting) body!!