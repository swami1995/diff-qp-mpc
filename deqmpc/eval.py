import sys
import os
import time
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, project_dir)
from torch.utils.tensorboard import SummaryWriter
import policies
from policies import *
from datagen import *
from rex_quadrotor import RexQuadrotor
from my_envs.cartpole import CartpoleEnv
from envs import PendulumEnv, IntegratorEnv
from envs_v1 import OneLinkCartpoleEnv
import ipdb
import qpth.qp_wrapper as mpc
import utils, noise_utils
import math
import numpy as np
import torch
import torch.autograd as autograd


torch.set_default_device('cuda')
np.set_printoptions(precision=4, suppress=True)
# import tensorboard from pytorch

# example task : hard pendulum with weird coordinates to make sure direct target tracking is difficult


def seeding(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    # expt = "deqmpc_cp1_Q_hdim256_Qreg0.05_T5_bsz200_deq_iter6"  #TODO: can be argument
    expt = "deqmpc_cartpole_5k_noineq_conditer_L1simplefix_cond_expand4_gcn_nodetach_again_T5_bsz200_deq_iter6_hdim256"
    args_file = "./logs/" + expt + "/args"
    model_file = "./model/" + expt

    args = torch.load(args_file)
    print(args)
    args.bsz = 1  #TODO: can be argument

    seeding(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device if args.device is None else args.device
    kwargs = {"dtype": torch.float64,
              "device": args.device, "requires_grad": False}

    if args.env == "pendulum":
        env = PendulumEnv(stabilization=False)
        gt_trajs = get_gt_data(args, env, "sac")
    elif args.env == "integrator":
        env = IntegratorEnv()
        gt_trajs = get_gt_data(args, env, "mpc")
    elif args.env == "rexquadrotor":
        env = RexQuadrotor(bsz=args.bsz, device=args.device)
        gt_trajs = get_gt_data(args, env, "cgac")
    elif args.env == "pendulum_stabilize":
        env = PendulumEnv(stabilization=True)
        gt_trajs = get_gt_data(args, env, "sac")
    elif args.env == "cartpole-v0":
        env = OneLinkCartpoleEnv(
            bsz=args.bsz, max_steps=args.T, stabilization=False)
        gt_trajs = get_gt_data(args, env, "cgac")
    elif args.env == "cartpole1link":
        env = CartpoleEnv(nx=4, dt=0.05, stabilization=False, kwargs=kwargs)
        gt_trajs = get_gt_data(args, env, "cgac")
    elif args.env == "cartpole2link":
        env = CartpoleEnv(nx=6, dt=0.03, stabilization=False, kwargs=kwargs)
        gt_trajs = get_gt_data(args, env, "cgac")
    else:
        raise NotImplementedError

    gt_trajs = merge_gt_data(gt_trajs)
    if args.deq:
        policy = DEQMPCPolicy(args, env).to(args.device)
        # policy = DEQMPCPolicyHistory(args, env).to(args.device)
        # policy = DEQMPCPolicyFeedback(args, env).to(args.device)
        # policy = DEQMPCPolicyQ(args, env).to(args.device)
    else:
        # policy = NNMPCPolicy(args, env).to(args.device)
        policy = NNPolicy(args, env).to(args.device)
    policy.load_state_dict(torch.load(model_file))
    eval_policy(args, env, policy, gt_trajs)


def eval_policy(args, env, policy, gt_trajs):
    policy.eval()
    torch.no_grad()

    ## THESE ARE NOT USED TECHNICALLY
    # sample bsz random trajectories from gt_trajs and a random time step for each
    args.T = 200
    traj_sample = sample_trajectory(gt_trajs, args.bsz, args.H, args.T)
    traj_sample = {k: v.to(args.device) for k, v in traj_sample.items()}

    if args.env == "pendulum":
        traj_sample["state"] = utils.unnormalize_states_pendulum(
            traj_sample["state"])
        traj_sample["obs"] = utils.unnormalize_states_pendulum(traj_sample["obs"])
    elif args.env == "cartpole1link" or args.env == "cartpole2link":
        traj_sample["state"] = utils.unnormalize_states_cartpole_nlink(
            traj_sample["state"])
        traj_sample["obs"] = utils.unnormalize_states_pendulum(traj_sample["obs"])

    gt_obs = traj_sample["obs"]
    noisy_obs = noise_utils.corrupt_observation(
        gt_obs, args.data_noise_type, args.data_noise_std, args.data_noise_mean)
    if args.H == 1:
        obs_in = noisy_obs.squeeze(1)
    else:
        obs_in = noisy_obs
    
    gt_actions = traj_sample["action"]
    gt_states = traj_sample["state"]
    gt_mask = traj_sample["mask"]
    ## 

    # initial state
    # ipdb.set_trace()
    # state = gt_states[:, 0, :]
    state = env.reset(bsz=args.bsz)
    # state = torch.tensor([[0., 0., 0., 0.]], device=args.device)  
    # state = state.repeat(args.bsz, 1)

    # high = np.array([np.pi, 1])
    # state = torch.tensor([np.random.uniform(low=-high, high=high)], dtype=torch.float32)

    # history of size bsz x N x nx
    state_hist = state[:,None,:]
    input_hist = torch.tensor([])

    NRUNS = 200
    # ipdb.set_trace()
    for i in range(NRUNS):      
        obs_in = state.clone()
        obs_in = env.state_clip(obs_in)

        if args.deq:
            policy_out = policy(obs_in, gt_states, gt_actions,
                                gt_mask, qp_solve=args.qp_solve, lastqp_solve=args.lastqp_solve)
        else:
            raise NotImplementedError
        
        if args.qp_solve or args.lastqp_solve:
            nominal_state_net, nominal_state, nominal_action = policy_out["trajs"][-1]      
            u = nominal_action[:, 0, :]
        elif args.bc:
            u = policy_out["actions"][-1].squeeze(1) 
        # ipdb.set_trace()
        u = gt_actions[:, i, :]
        u = env.action_clip(u)
        # print("nominal states\n", nominal_state)
        # print("nominal actions\n", nominal_action)
        # ipdb.set_trace()
        state = env.dynamics(state.to(torch.float64), u.to(torch.float64)).to(torch.float32)
        state = env.state_clip(state)
        state_hist = torch.cat((state_hist, state[:,None,:]), dim=1)
        input_hist = torch.cat((input_hist, u[:,None,:]), dim=1)
        # print(x_ref)
        # ipdb.set_trace()
        # print(torch.norm(state[4] - gt_states[4, i+1, :]))
        
    
    # print(state_hist[:,-1,:])
    ipdb.set_trace()

    #TODO: Compute some metrics

    # plt.figure()
    # plt.plot(theta, label='theta', color='red', linewidth=2.0, linestyle='-')
    # plt.plot(theta_dot, label='theta_dot', color='blue', linewidth=2.0, linestyle='-')
    # plt.plot(torque, label='torque', color='green', linewidth=2.0, linestyle='--')
    # plt.legend()
    # # plt.ylim(-env.max_acc*1.5, env.max_acc*1.5)
    # plt.show()        

    # utils.animate_pendulum(env, theta, torque)
    # utils.animate_integrator(env, theta, torque)
    # utils.anime_cartpole1(env, pos)


if __name__ == "__main__":
    main()