import math
import time

import numpy as np
import torch
import torch.autograd as autograd

import sys

sys.path.insert(0, "/home/khai/diff-qp-mpc")
import qpth.qp_wrapper as mpc
import ipdb
from envs.cartpole import CartpoleEnv
from datagen import get_gt_data, merge_gt_data, sample_trajectory
import matplotlib.pyplot as plt
from policies import Tracking_MPC
import utils

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
    parser.add_argument("--solver_type", type=str, default="al")
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    kwargs = {"dtype": torch.float64, "device": "cpu", "requires_grad": False}
    nx = 6
    dt = 0.05
    env = CartpoleEnv(nx=nx, dt=dt, stabilization=False, kwargs=kwargs)

    # enum of mode of operation
    # 0: test uncontrolled dynamics
    # 1: test ground truth trajectory
    # 2: test controlled dynamics
    mode = 0

    # test uncontrolled dynamics
    if mode == 0:
        state = torch.tensor([[0.0, np.pi, 0.0, 0.0, 15.0, 15.0]], **kwargs)
        desired_state = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], **kwargs)
        state_hist = state
        torque = torch.tensor([[10.0]], **kwargs)
        # Kinf = -torch.tensor([[8.535, 231.96, 954.696, 31.6, 157.97, 123.608]], **kwargs)
        for i in range(200):
            # torque = -Kinf @ (state - desired_state).T
            state = env.dynamics(state, torque)
            state_hist = torch.cat((state_hist, state), dim=0)
        theta = state_hist[:, : env.nq]
        theta_dot = state_hist[:, env.nq :]

        plt.figure()
        print(state_hist.shape)
        plt.plot(utils.to_numpy(theta), label="theta", linewidth=2.0, linestyle="-")
        # plt.plot(theta_dot, label='theta_dot', color='blue', linewidth=2.0, linestyle='-')
        # plt.legend()
        plt.show()

    # ground truth trajectory
    if mode == 1:
        gt_trajs = get_gt_data(args, env, "sac")
        idx = 10
        theta = [item[0][0] for item in gt_trajs[idx]]
        theta_dot = [item[0][1] for item in gt_trajs[idx]]
        torque = [item[1][0] for item in gt_trajs[idx]]
        plt.plot(theta, label="theta", color="red", linewidth=2.0, linestyle="-")
        plt.plot(
            theta_dot, label="theta_dot", color="blue", linewidth=2.0, linestyle="-"
        )
        plt.plot(torque, label="torque", color="green", linewidth=2.0, linestyle="--")
        plt.legend()
        plt.show()

    # test controlled dynamics
    torch.no_grad()
    if mode == 2:
        args.T = 100
        args.device = "cpu"
        args.qp_iter = 1
        args.eps = 1e-2
        args.warm_start = True
        args.bsz = 1
        args.Q = torch.Tensor([10.0, 10.0, 10, 1.0, 1.0, 1.0])
        args.R = torch.Tensor([1.0])
        # args.solver_type == "al"

        # test controlled dynamics
        state = torch.tensor([[0.0, np.pi+np.pi, 0.1, 0.0, 0.0, 0.0]], **kwargs)
        # high = np.array([np.pi, 1])
        # state = torch.tensor([np.random.uniform(low=-high, high=high)], dtype=torch.float32)

        state_hist = state
        torque_hist = [0.0]

        tracking_mpc = Tracking_MPC(args, env)
        torch.no_grad()
        for i in range(170):
            x_ref = torch.zeros((args.bsz, args.T, 6), **kwargs)
            u_ref = torch.zeros((args.bsz, args.T, 1), **kwargs)
            xu_ref = torch.zeros((args.bsz, args.T, 7), **kwargs)
            # ipdb.set_trace()
            nominal_states, nominal_action = tracking_mpc(state, xu_ref, x_ref, u_ref)
            print("reference states\n", x_ref)
            print("nominal states\n", nominal_states)
            u = nominal_action[0, :, 0]

            state = env.dynamics(state, u)
            state_hist = torch.cat((state_hist, state), dim=0)
            # ipdb.set_trace()
            torque_hist.append(utils.to_numpy(u)[0])
            # print(x_ref)

        theta = state_hist[:, 0].detach().numpy()
        theta_dot = state_hist[:, 1].detach().numpy()
        # ipdb.set_trace()
        torque = torque_hist

        plt.figure()
        plt.plot(theta, label="theta", color="red", linewidth=2.0, linestyle="-")
        plt.plot(
            theta_dot, label="theta_dot", color="blue", linewidth=2.0, linestyle="-"
        )
        plt.plot(torque, label="torque", color="green", linewidth=2.0, linestyle="--")
        plt.legend()
        # plt.ylim(-env.max_acc*1.5, env.max_acc*1.5)
        plt.show()

    # utils.animate_pendulum(env, theta, torque)
    # utils.animate_integrator(env, theta, torque)
    utils.animate_cartpole2(utils.to_numpy(state_hist.T))


if __name__ == "__main__":
    main()
