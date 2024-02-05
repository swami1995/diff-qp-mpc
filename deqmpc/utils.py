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