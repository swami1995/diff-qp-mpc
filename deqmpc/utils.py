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