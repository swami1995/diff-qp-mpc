import math
import time

import numpy as np
import torch
import torch.autograd as autograd
import sys
sys.path.insert(0, "/home/khai/diff-qp-mpc")
import utils
from policies import NNMPCPolicy, DEQPolicy, DEQMPCPolicy, NNPolicy, Tracking_MPC
import matplotlib.pyplot as plt
from datagen import get_gt_data, merge_gt_data, sample_trajectory
from my_envs.cartpole import CartpoleEnv
import ipdb
import qpth.qp_wrapper as mpc



# example task : hard pendulum with weird coordinates to make sure direct target tracking is difficult


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="pendulum")
    parser.add_argument("--np", type=int, default=2)  # TODO configurations
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument('--dt', type=float, default=0.02)
    parser.add_argument("--qp_iter", type=int, default=1)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warm_start", type=bool, default=True)
    parser.add_argument("--bsz", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--deq", action="store_true")
    parser.add_argument("--hdim", type=int, default=512)
    parser.add_argument("--deq_iter", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--layer_type", type=str, default='mlp')
    parser.add_argument("--kernel_width", type=int, default=3)
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--lastqp_solve", action="store_true")
    parser.add_argument("--qp_solve", action="store_true")
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--solver_type", type=str, default='al')
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--dtype", type=str, default="double")
    parser.add_argument("--ckpt", type=str, default="bc_sac_pen")

    args = parser.parse_args()
    args.device = "cpu"
    kwargs = {"dtype": torch.float64 if args.dtype ==
              "double" else torch.float32, "device": args.device, "requires_grad": False}
    nq = 2
    nx = nq*2
    nu = 1
    env = CartpoleEnv(nx=nx, dt=args.dt, stabilization=False, kwargs=kwargs)

    # enum of mode of operation
    # 0: test uncontrolled dynamics
    # 1: test ground truth trajectory
    # 2: test controlled dynamics
    mode = 2

    # test uncontrolled dynamics
    if mode == 0:
        state = torch.tensor([[0.0, np.pi, 0.0, 0.0]], **kwargs)
        desired_state = torch.tensor([[0.0, 0.0, 0.0, 0.0]], **kwargs)
        state_hist = state
        torque = torch.tensor([[20.0]], **kwargs)
        # torque = torch.randn((1, 1), **kwargs) * 5
        for i in range(200):
            # torque = -Kinf @ (state - desired_state).T
            # torque = torch.randn((1, 1), **kwargs) * 15
            state = env.dynamics(state, torque)
            state_hist = torch.cat((state_hist, state), dim=0)
        theta = state_hist[:, : env.nq]
        theta_dot = state_hist[:, env.nq:]
        # ipdb.set_trace()

        # plt.figure()
        # print(state_hist.shape)
        # plt.plot(utils.to_numpy(theta), label="theta",
        #          linewidth=2.0, linestyle="-")
        # # plt.plot(theta_dot, label='theta_dot', color='blue', linewidth=2.0, linestyle='-')
        # # plt.legend()
        # plt.show()

    # ground truth trajectory
    if mode == 1:
        gt_trajs = get_gt_data(args, env, "sac")
        idx = 10
        theta = [item[0][0] for item in gt_trajs[idx]]
        theta_dot = [item[0][1] for item in gt_trajs[idx]]
        torque = [item[1][0] for item in gt_trajs[idx]]
        plt.plot(theta, label="theta", color="red",
                 linewidth=2.0, linestyle="-")
        plt.plot(
            theta_dot, label="theta_dot", color="blue", linewidth=2.0, linestyle="-"
        )
        plt.plot(torque, label="torque", color="green",
                 linewidth=2.0, linestyle="--")
        plt.legend()
        plt.show()

    # test controlled dynamics
    torch.no_grad()
    if mode == 2:
        args.warm_start = True
        args.bsz = 1
        args.Q = 10*torch.Tensor([1.0, 10.0, 1, 1.0])
        args.R = torch.Tensor([0.1])
        # args.solver_type = "al"

        # test controlled dynamics
        state = torch.tensor([[0., 0.2, 0.0, 0.0]], **kwargs)
        # high = np.array([np.pi, 1])
        # state = torch.tensor([np.random.uniform(low=-high, high=high)], dtype=torch.float32)

        state_hist = state
        torque_hist = [0.0]

        tracking_mpc = Tracking_MPC(args, env)

        torch.no_grad()
        # for i in range(170):
        x_ref = torch.zeros((args.bsz, args.T, nx), **kwargs)
        u_ref = torch.rand((args.bsz, args.T, nu), **kwargs)*0.01
        xu_ref = torch.cat((x_ref, u_ref), dim=-1)
        if (args.solver_type == "al"):
            tracking_mpc.reinitialize(
                x_ref, torch.ones(args.bsz, args.T, 1, **kwargs))

        nominal_states, nominal_action = tracking_mpc(
            state, xu_ref, x_ref, u_ref)
        state_hist = nominal_states.squeeze(0)  
        # print("nominal states\n", nominal_states)
        print("nominal action\n", nominal_action)

        # state_hist = state
        # for i in range(args.T):
        #     state = env.dynamics(state, nominal_action[:, i])
        #     # ipdb.set_trace()
        #     state_hist = torch.cat((state_hist, state), dim=0)
        # print(state)

        # state = env.dynamics(state, u)
        # state_hist = torch.cat((state_hist, state), dim=0)
        # # ipdb.set_trace()
        # torque_hist.append(utils.to_numpy(u)[0])
        # # print(x_ref)

        # theta = state_hist[:, 0].detach().numpy()
        # theta_dot = state_hist[:, 1].detach().numpy()
        # # ipdb.set_trace()
        # torque = torque_hist

        # plt.figure()
        # plt.plot(theta, label="theta", color="red", linewidth=2.0, linestyle="-")
        # plt.plot(
        #     theta_dot, label="theta_dot", color="blue", linewidth=2.0, linestyle="-"
        # )
        # plt.plot(torque, label="torque", color="green", linewidth=2.0, linestyle="--")
        # plt.legend()
        # # plt.ylim(-env.max_acc*1.5, env.max_acc*1.5)
        # plt.show()

    # utils.animate_pendulum(env, theta, torque)
    # utils.animate_integrator(env, theta, torque)
    utils.animate_cartpole(utils.to_numpy(state_hist.T), nq)
    # utils.anime_cartpole1(utils.to_numpy(state_hist.T))


if __name__ == "__main__":
    main()
