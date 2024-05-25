import math
import time

import numpy as np
import torch
import torch.autograd as autograd

import sys

# sys.path.insert(0, "/home/khai/diff-qp-mpc")
import os
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, project_dir)
import qpth.qp_wrapper as mpc
import ipdb
from rexquad_utils import rk4, deg2rad, Spaces, Spaces_np, w2pdotkinematics_mrp, quat2mrp, euler_to_quaternion, mrp2quat, quatrot, mrp2rot
from flying_cartpole2d import FlyingCartpole
from rex_quadrotor import RexQuadrotor
from datagen import get_gt_data, merge_gt_data, sample_trajectory
import matplotlib.pyplot as plt
from policies import NNMPCPolicy, DEQPolicy, DEQMPCPolicy, NNPolicy, Tracking_MPC
import utils
import pickle
## example task : hard pendulum with weird coordinates to make sure direct target tracking is difficult


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="pendulum")
    parser.add_argument("--np", type=int, default=3)  # TODO configurations
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument('--dt', type=float, default=0.05)
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
    kwargs = {"dtype": torch.float64 if args.dtype == "double" else torch.float32, "device": args.device, "requires_grad": False}
    nq = 7
    nx = nq*2
    nu = 4
    env = FlyingCartpole(bsz=args.bsz, device=args.device)
    # env = RexQuadrotor(bsz=args.bsz, device=args.device)

    # enum of mode of operation
    # 0: test uncontrolled dynamics
    # 1: test ground truth trajectory
    # 2: test controlled dynamics
    mode = 2

    # test controlled dynamics
    torch.no_grad()
    if mode == 2:
        args.warm_start = True
        args.bsz = 1
        args.Q = 1*torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        # args.Q = 100*torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        args.R = 1e-6*torch.Tensor([1, 1, 1, 1])
        # args.solver_type = "al"

        # test controlled dynamics
        # state = torch.tensor([[0.0, 0.1, 0.0, 0.0, 0.0, 0.0]], **kwargs)
        # state = torch.tensor([.0,1.0,1.0,deg2rad(10.0),deg2rad(0.0),deg2rad(0.0),0,0.,0.,0.,0.,0.], **kwargs).unsqueeze(0)
        # state = torch.tensor([5.0,5.0,5.0,deg2rad(45.0),deg2rad(45.0),deg2rad(45.0),np.pi-0.5,1.0,1.0,1.0,1., 1.0, 1.0, 1.0], **kwargs).unsqueeze(0)
        state = torch.tensor([5.0,5.0,5.0,deg2rad(45.0),deg2rad(45.0),deg2rad(45.0),np.pi+0.5,1.0,1.0,1.0,1.0,-1.0,1.0,-1.0], **kwargs).unsqueeze(0)
        state = torch.cat([state[:,:3], quat2mrp(euler_to_quaternion(state[:, 3:6])), state[:, 6:]], dim=-1).repeat(args.bsz, 1)
        # state = torch.rand((args.bsz, nx), **kwargs)
        # high = np.array([1, np.pi, np.pi, 1, 1, 1])
        # state = torch.tensor([np.random.uniform(low=-high, high=high)], dtype=torch.float32)

        state_hist = state
        torque_hist = [0.0]

        tracking_mpc = Tracking_MPC(args, env)
        
        torch.no_grad()

        x_init = torch.rand((args.bsz, args.T, nx), **kwargs)
        x_ref = torch.zeros(args.bsz, args.T, nx, **kwargs)
        for i in range(args.T):
            x_init[:, i, :] = state
            x_ref[:, i, 6] = np.pi        
        u_ref = torch.zeros(args.bsz, args.T, nu, **kwargs)
        xu_ref = torch.cat((x_ref, u_ref), dim=-1)
        if (args.solver_type == "al"):
            tracking_mpc.reinitialize(x_init, torch.ones(args.bsz, args.T, 1, **kwargs))
            
        nominal_states, nominal_action = tracking_mpc(
            state, xu_ref, x_ref, u_ref, al_iters=20)
        state_hist = nominal_states[0]
        print("nominal states\n", nominal_states)
        print("nominal action\n", nominal_action.view(-1))
        ipdb.set_trace()

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

if __name__ == "__main__":
    main()
