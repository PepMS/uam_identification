# Some data plots
plot_step = 30 # To lighten the data

# Rotor Velocity
plt.figure()
plt.plot(np.array(rotor_ts[:-1:plot_step]), rotor_rads[:-1:plot_step])
plt.title('Rotation velocity [rad/s]')
plt.legend(['rotor 1', 'rotor 2', 'rotor 3', 'rotor 4'])

# Attitude quaternion
plt.figure()
q_plot = [q.coeffs() for q in q_nwu_flus]
plt.plot(np.array(R_ts[:-1:plot_step]), q_plot[:-1:plot_step])
plt.title('Quaternion')
plt.legend(['x', 'y', 'z', 'w'])

# Attitude RPYs
plt.figure()
plt.plot(np.array(R_ts[:-1:plot_step]), rpys[:-1:plot_step])
plt.title('Angles. [rad]')
plt.legend(['R', 'P', 'Y'])

# Linear acceleration
plt.figure()
plt.plot(np.array(acc_ts[:-1:plot_step]), acc_flus[:-1:plot_step])
plt.title('Accelerations. FLU frame [m/s^2]')
plt.legend(['x', 'y', 'z'])

# Angular velocity
plt.figure()
plt.plot(np.array(ang_vel_ts[:-1:plot_step]), ang_vel_flus[:-1:plot_step])
plt.title('Angular velocity. FLU frame [rad/s]')
plt.legend(['x', 'y', 'z'])

plt.show()
