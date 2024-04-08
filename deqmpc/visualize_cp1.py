import math
import time

import numpy as np
import torch
import torch.autograd as autograd

import sys
sys.path.insert(0, '/home/swaminathan/Workspace/qpth/')
import qpth.qp_wrapper as mpc
import ipdb
from envs_v1 import OneLinkCartpoleEnv
from deqmpc.datagen_cp1 import get_gt_data, merge_gt_data, sample_trajectory
import matplotlib.pyplot as plt
from policies import NNMPCPolicy, DEQPolicy, DEQMPCPolicy, NNPolicy, Tracking_MPC
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
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    env = OneLinkCartpoleEnv(stabilization=False)

    # enum of mode of operation
    # 0: test uncontrolled dynamics
    # 1: test ground truth trajectory
    # 2: test controlled dynamics
    mode = 0

    # test uncontrolled dynamics
    if mode == 0:
        state = torch.Tensor([[1.0, np.pi, 0.0, 0.0]])
        desired_state = torch.Tensor([[0.0, np.pi, 0.0, 0.0]])
        state_hist = state
        torque = torch.Tensor([[10.0]])
        Kinf = torch.Tensor([[-1.246, 15.896, -0.858, 2.598]])
        for i in range(2):        
            # torque = -1*Kinf @ (state - desired_state).T
            print(state, torque)
            state = env.dynamics(state, torque)
            state_hist = torch.cat((state_hist, state), dim=0)
        pos = state_hist[:, :env.np]
        vel = state_hist[:, env.np:]

        plt.figure()
        print(state_hist.shape)
        plt.plot(pos, label='pos', linewidth=2.0, linestyle='-')
        # plt.plot(theta_dot, label='theta_dot', color='blue', linewidth=2.0, linestyle='-')
        # plt.legend()
        plt.show()

    # ground truth trajectory
    if mode == 1:
        gt_trajs = get_gt_data(args, env, "mpc")
        idx = 0
        pos = ([item[0][0:env.np] for item in gt_trajs[idx]])
        # import ipdb; ipdb.set_trace()
        vel = [item[0][env.np:] for item in gt_trajs[idx]]
        torque = [item[1][0] for item in gt_trajs[idx]]
        pos = np.array(pos)
        plt.plot(pos, label='theta', color='red', linewidth=2.0, linestyle='-')
        # plt.plot(theta_dot, label='theta_dot', color='blue', linewidth=2.0, linestyle='-')
        plt.plot(torque, label='torque', color='green', linewidth=2.0, linestyle='--')
        plt.legend()
        plt.show()

    # test controlled dynamics
    torch.no_grad()
    if mode == 2:
        args = torch.load("./model/bc_sac_pen_args")
        args.device = "cpu"
        args.bsz = 1
        args.Q = torch.Tensor([100.0, 10.0])
        args.R = torch.Tensor([1.0])
        policy = NNPolicy(args, env)
        policy.load_state_dict(torch.load("./model/bc_sac_pen"))
        policy.eval()
        # test controlled dynamics
        state = torch.Tensor([[1.8, -0.4]])
        # high = np.array([np.pi, 1])
        # state = torch.tensor([np.random.uniform(low=-high, high=high)], dtype=torch.float32)

        state_hist = state
        torque_hist = [0.0]

        if policy.output_type == 0 or policy.output_type == 2:
            for i in range(170):        
                _, action = policy(state)
                state = env.dynamics(state, action[:, 0, 0])
                state_hist = torch.cat((state_hist, state), dim=0)
                torque_hist.append(action[:, 0, 0].detach().numpy()[0])

        if policy.output_type == 1 or policy.output_type == 3:
            tracking_mpc = Tracking_MPC(args, env)            
            torch.no_grad()
            for i in range(170):        
                x_ref, _ = policy(state)
                xu_ref = torch.cat(
                    [x_ref, torch.zeros_like(x_ref[..., :1])], dim=-1
                ).transpose(0, 1)
                # ipdb.set_trace()
                nominal_states, nominal_action = tracking_mpc(state, xu_ref)
                print("reference states\n", x_ref)
                print("nominal states\n", nominal_states)            
                u = nominal_action[0, :, 0]

                state = env.dynamics(state, u)
                state_hist = torch.cat((state_hist, state), dim=0)
                # ipdb.set_trace()
                torque_hist.append(u.detach().numpy()[0])
                # print(x_ref)
        
        theta = state_hist[:, 0].detach().numpy()
        theta_dot = state_hist[:, 1].detach().numpy()
        # ipdb.set_trace()
        torque = torque_hist

        plt.figure()
        plt.plot(theta, label='theta', color='red', linewidth=2.0, linestyle='-')
        plt.plot(theta_dot, label='theta_dot', color='blue', linewidth=2.0, linestyle='-')
        plt.plot(torque, label='torque', color='green', linewidth=2.0, linestyle='--')
        plt.legend()
        # plt.ylim(-env.max_acc*1.5, env.max_acc*1.5)
        plt.show()        

    # utils.animate_pendulum(env, theta, torque)
    # utils.animate_integrator(env, theta, torque)
    utils.anime_cartpole1(env, pos)

if __name__ == "__main__":
    main()
