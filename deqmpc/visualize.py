import math
import time

import numpy as np
import torch
import torch.autograd as autograd
import qpth.qp_wrapper as mpc
import ipdb
from envs import PendulumEnv, PendulumDynamics
from datagen import get_gt_data, merge_gt_data, sample_trajectory
import matplotlib.pyplot as plt

## example task : hard pendulum with weird coordinates to make sure direct target tracking is difficult

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--np", type=int, default=1)
    parser.add_argument("--T", type=int, default=5)
    # parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument("--qp_iter", type=int, default=10)
    parser.add_argument("--eps", type=float, default=1e-2)
    parser.add_argument("--warm_start", type=bool, default=True)
    parser.add_argument("--bsz", type=int, default=80)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    env = PendulumEnv(stabilization=False)

    # test uncontrolled dynamics
    if 0:
        state = torch.Tensor([[1.5, 0]])
        state_hist = state
        for i in range(200):        
            state = env.dynamics(state, torch.Tensor([0.0]))
            state_hist = torch.cat((state_hist, state), dim=0)
        theta = state_hist[:, 0]
        theta_dot = state_hist[:, 1]

        # plt.figure()
        # print(state_hist.shape)
        # plt.plot(theta, label='theta', color='red', linewidth=2.0, linestyle='-')
        # plt.plot(theta_dot, label='theta_dot', color='blue', linewidth=2.0, linestyle='-')
        # plt.legend()

    # ground truth trajectory
    else:
        gt_trajs = get_gt_data(args, env, "mpc")
        idx = 0
        theta = [item[0][0] for item in gt_trajs[idx]]
        theta_dot = [item[0][1] for item in gt_trajs[idx]]
        torque = [item[1][0] for item in gt_trajs[idx]]
        plt.plot(theta, label='theta', color='red', linewidth=2.0, linestyle='-')
        plt.plot(theta_dot, label='theta_dot', color='blue', linewidth=2.0, linestyle='-')
        plt.plot(torque, label='torque', color='green', linewidth=2.0, linestyle='--')
        plt.legend()
        plt.show()

    from matplotlib.animation import FuncAnimation
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

if __name__ == "__main__":
    main()
