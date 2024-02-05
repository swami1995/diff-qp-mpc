import numpy as np
import matplotlib.pyplot as plt

def cartpole1_dynamics(state, t, u, params):
    M, m1, l1, g = params["M"], params["m1"], params["l1"], params["g"]
    x, theta1, dx, dtheta1 = state

    # cos_theta0 = np.cos(theta0)
    
    # M = np.array([
    #     [1, 0, 0, 0],
    #     [0, 1, 0, 0],
    #     [0, 0, m0 + m0, m0 * r0 * cos_theta0],
    #     [0, 0, m0 * r0 * cos_theta0, I0 + m0 * r0**2]
    # ])
    M = np.array([
        [1, 0, 0, 0,],
        [0, 1, 0, 0,],
        [0, 0, m1, 0,],
        [0, 0, 0, m1 * l1**2,]

def simulate_cartpole(initial_state, total_time, control_input, params):
    """
    Simulate the 2-link cartpole system using Euler's method.
    Args:
        initial_state (list): Initial state [x, theta1, theta2, dx, dtheta1, dtheta2]
        total_time (float): Total simulation time
        control_input (function): Control input as a function of time
        params (dict): System parameters (M, m1, m2, l1, l2, g)

    Returns:
        time (numpy array): Time array
        states (numpy array): State trajectory over time
    """
    dt = 0.01  # Time step
    num_steps = int(total_time / dt)

    time = np.linspace(0, total_time, num_steps)
    states = np.zeros((num_steps, len(initial_state)))
    states[0, :] = initial_state

    for i in range(1, num_steps):
        u = control_input(time[i])
        derivatives = cartpole_dynamics(states[i - 1, :], time[i - 1], u, params)
        states[i, :] = states[i - 1, :] + np.array(derivatives) * dt

    return time, states

# Define system parameters
params = {'M': 1.0, 'm1': 0.1, 'm2': 0.1, 'l1': 1.0, 'l2': 1.0, 'g': 9.8}

# Define initial state [x, theta1, theta2, dx, dtheta1, dtheta2]
initial_state = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

# Define a simple control input function (you can replace this with your own controller)
def control_input(t):
    return 0.0

# Simulate the system
total_time = 10.0
time, states = simulate_cartpole(initial_state, total_time, control_input, params)

# Plot the results
plt.plot(time, states[:, 0], label='Cart Position')
plt.plot(time, states[:, 1], label='Theta1')
plt.plot(time, states[:, 2], label='Theta2')
plt.xlabel('Time (s)')
plt.ylabel('State Variables')
plt.legend()
plt.show()
