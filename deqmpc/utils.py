import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_pendulum(env, theta, torque):
    """
    Animate the pendulum using matplotlib
    Args:
        env: pendulum environment
        theta: list of pendulum angles
        torque: list of pendulum torques
    """

    # Animation function
    def update(frame):
        ax.clear()
        
        # Set up pendulum parameters
        length = 1.0  # Length of the pendulum (meters)
        
        # Calculate pendulum position
        angle = theta[frame] 
        
        # Plot pendulum
        x = [0, -length * np.sin(angle)]
        y = [0, length * np.cos(angle)]
        ax.plot(x, y, marker='o', markersize=10, color='blue', linewidth=4)
        ax.arrow(0, -1, torque[frame]/env.max_torque, 0, color='green', width=0.05)
        
        # Set plot limits
        ax.set_xlim(-length*1.5, length*1.5)
        ax.set_ylim(-length*1.5, length*1.5)
        
        # Set plot aspect ratio to be equal
        ax.set_aspect('equal')

    # Set up the plot
    fig, ax = plt.subplots()
    ani = FuncAnimation(fig, update, frames=len(theta), interval=30, repeat=True)

    plt.title('Simple Pendulum Animation')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')

    # Display the animation
    plt.show()

def animate_integrator(env, pos, acc):
    # animate 2d position and acceleration as arrow
    def update(frame):
        ax.clear()
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.arrow(0, 0, pos[frame, 0], pos[frame, 1], color='blue', width=0.05)
        ax.arrow(pos[frame, 0], pos[frame, 1], acc[frame, 0], acc[frame, 1], color='green', width=0.05)
    fig, ax = plt.subplots()
    ani = FuncAnimation(fig, update, frames=len(pos), interval=30, repeat=True)
    plt.title('Integrator Animation')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.show()
    
def anime_cartpole1(env, pos, force=0):
    # animate cartpole
    def update(frame):
        ax.clear()
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.plot([pos[frame, 0], pos[frame, 0] + np.sin(pos[frame, 1])], [0, -np.cos(pos[frame, 1])], color='blue', linewidth=4)
        if (force):
            ax.arrow(pos[frame, 0], 0, force[frame], 0, color='green', width=0.05)
    fig, ax = plt.subplots()
    ani = FuncAnimation(fig, update, frames=len(pos), interval=30, repeat=True)
    plt.title('Cartpole Animation')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.show()

####################
import time    
def dpc_draw(X):
    # get a random starting state between min state and max state
    p = {"r_1": 1.0, "r_2": 1.0}

    x0 = X[:,1];  # first state at k = 1

    # this function is defined below
    [p_c, p_1, p_2] = dpc_endpositions(x0[1], x0[2], x0[3], p)

    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')
    print(p)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    # plt.show()
    # plt.draw()
    # plt.pause(0.5)

    # timer_handle = plt.text(-0.3, x_max[0], '0.00 s', fontsize=15);
    cart_handle, = plt.plot(p_c[0], p_c[1], 'ks', markersize=20, linewidth=3);
    pole_one_handle, = plt.plot([p_c[0], p_1[0]], [p_c[1], p_1[1]], color=np.array([38,124,185])/255, linewidth=8);
    pole_two_handle, = plt.plot([p_1[0], p_2[0]], [p_1[1], p_2[1]], color=np.array([38,124,185])/255, linewidth=8);

    joint_zero_handle, = plt.plot(p_c[0], p_c[1], 'ko', markersize=5)
    joint_one_handle, = plt.plot(p_1[0], p_1[1], 'ko', markersize=5)
    joint_two_handle, = plt.plot(p_2[0], p_2[1], 'ko', markersize=5)

    for k in range(0, X.shape[1]):
        tic = time.time()
        x = X[:,k]

        [p_c, p_1, p_2] = dpc_endpositions(x[0], x[1], x[2], p)

        # timer_handle.set_text('{:.2f} s'.format(tdata[k]))

        cart_handle.set_data(x[0], 0)

        pole_one_handle.set_data([p_c[0], p_1[0]], [p_c[1], p_1[1]])
        pole_two_handle.set_data([p_1[0], p_2[0]], [p_1[1], p_2[1]])

        joint_zero_handle.set_data(p_c[0], p_c[1]);
        joint_one_handle.set_data(p_1[0], p_1[1]);
        joint_two_handle.set_data(p_2[0], p_2[1]);

        time.sleep(np.max([0.1, 0]))
        plt.pause(0.0001)
    plt.close(fig)

def dpc_endpositions(q_0, q_1, q_2, p):
    # Returns the positions of cart, first joint, and second joint
    # to draw the black circles
    p_c = np.array([q_0, 0]);
    p_1 = p_c + p["r_1"] * np.array([np.cos(q_1), np.sin(q_1)])
    p_2 = p_c + p["r_1"] * np.array([np.cos(q_1), np.sin(q_1)]) + p["r_2"] * np.array([np.cos(q_1+q_2), np.sin(q_1+q_2)]);
    return p_c, p_1, p_2