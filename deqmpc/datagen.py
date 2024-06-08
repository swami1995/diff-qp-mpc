import math
import time

import numpy as np
import torch
import torch.autograd as autograd

import sys
import os
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, project_dir)

# import qpth.qp_wrapper as mpc
import qpth.AL_mpc as mpc
# import qpth.sl1qp_mpc as mpc
import ipdb
import argparse
from envs import PendulumEnv, PendulumDynamics, IntegratorEnv, IntegratorDynamics
from rex_quadrotor import RexQuadrotor
from flying_cartpole2d import FlyingCartpole
from my_envs.cartpole import CartpoleEnv, Cartpole2linkEnv
from ppo_train import PPO, GaussianPolicy, CGACGaussianPolicy, CGACRunningMeanStd
from policies import *
import pickle
from rexquad_utils import rk4

class PendulumExpert:
    def __init__(self, env, type="mpc"):
        """
        Initialize the MPC controller with the necessary parameters.

        Args:
            env: The PendulumEnv environment.
            type: The type of controller to use. Can be 'mpc' or 'ppo' or 'sac'.
        """

        self.type = type

        if self.type == "mpc":
            self.T = 10
            self.goal_state = torch.Tensor([0.0, 0.0])
            self.goal_weights = torch.Tensor([10.0, 0.1])
            self.ctrl_penalty = 0.001
            self.mpc_eps = 1e-3
            self.linesearch_decay = 0.2
            self.max_linesearch_iter = 5
            self.nx = env.observation_space.shape[0]
            self.nu = env.action_space.shape[0]
            self.bsz = 2

            self.u_lower = torch.tensor(
                env.action_space.low, dtype=torch.float32
            ).double()
            self.u_upper = torch.tensor(
                env.action_space.high, dtype=torch.float32
            ).double()

            self.qp_iter = 1
            self.u_init = torch.zeros(self.T, self.bsz, self.nu).double()
            self.q = torch.cat(
                (self.goal_weights, self.ctrl_penalty * torch.ones(self.nu))
            )
            self.px = -torch.sqrt(self.goal_weights) * self.goal_state
            self.p = torch.cat((self.px, torch.zeros(self.nu)))
            self.Q = (
                torch.diag(self.q)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(self.T, self.bsz, 1, 1)
                .double()
            )
            self.p = self.p.unsqueeze(0).repeat(self.T, self.bsz, 1).double()

            self.ctrl = mpc.MPC(
                self.nx,
                self.nu,
                self.T,
                u_lower=self.u_lower,
                u_upper=self.u_upper,  # .double(),
                qp_iter=self.qp_iter,
                exit_unconverged=False,
                eps=1e-2,
                n_batch=self.bsz,
                backprop=False,
                verbose=0,
                u_init=self.u_init,  # .double(),
                grad_method=mpc.GradMethods.AUTO_DIFF,
                solver_type="dense",
                single_qp_solve=True,  # linear system
            )
            self.cost = mpc.QuadCost(self.Q, self.p)

    def optimize_action(self, state):
        """Solve the MPC problem for the given state."""
        # ipdb.set_trace()
        state = torch.cat([state, state], dim=0)
        nominal_states, nominal_actions = self.ctrl(
            state.double(), self.cost, env.dynamics
        )
        u = torch.clamp(nominal_actions[0], self.u_lower, self.u_upper)
        return u  # Return the first action in the optimal sequence

    def energy_shaping_action(self, state):
        """Compute the energy shaping action for the given state."""
        th = state[:, 0]
        # th = angle_normalize(th)
        thdot = state[:, 1]
        # ipdb.set_trace()
        E_tilde = 0.5 * thdot**2 - 10 * (1 + torch.cos(th + np.pi))
        u = -0.5 * thdot * E_tilde
        u = torch.clamp(u, self.u_lower, self.u_upper)
        return u.unsqueeze(1)


def get_pendulum_expert_traj_mpc(env, num_traj):
    """
    Get expert trajectories for pendulum environment using MPC for trajectory optimization.
    Args:
        env: The PendulumEnv environment.
        mpc_controller: The PendulumExpert.
        num_traj: Number of trajectories to save.
    Returns:
        A list of trajectories, each trajectory is a list of (state, action) tuples.
    """
    mpc_controller = PendulumExpert(env)
    trajectories = []
    for _ in range(num_traj):
        state = env.reset()  # Reset environment to a new initial state
        traj = []
        done = False
        while not done:
            action = mpc_controller.optimize_action(
                torch.tensor(state, dtype=torch.float32).view(1, -1)
            )
            # ipdb.set_trace()
            next_state, _, done, _ = env.step(action)
            traj.append((state, action.numpy()[0]))
            state = next_state[0]
            # ipdb.set_trace()
            # if len(traj) > 100:
            #     ipdb.set_trace()
            print(len(traj))
        print(f"Trajectory length: {len(traj)}")
        trajectories.append(traj)
    return trajectories


def get_pendulum_expert_traj_ppo(env, num_traj):
    """
    Get expert trajectories for pendulum environment using the saved PPO checkpoint."""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    lr = 0.0003
    gamma = 0.99
    K_epochs = 80
    eps_clip = 0.2
    has_continuous_action_space = True
    action_std = 0.45
    ppo_policy = PPO(
        state_dim,
        action_dim,
        lr,
        lr,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space,
        action_std,
    )
    ppo_policy.load("PPO_preTrained/Pendulum-v0/PPO_Pendulum-v0_0_0.pth")
    trajectories = []
    reward_trajs = []
    for _ in range(num_traj):
        state = env.reset()
        traj = []
        done = False
        reward_traj = 0
        while not done:
            action = (
                ppo_policy.policy.actor(
                    torch.tensor(state, dtype=torch.float32).to(
                        ppo_policy.policy.action_var.device
                    )
                )
                .detach()
                .cpu()
                .numpy()
            )
            next_state, reward, done, _ = env.step(action)
            traj.append((state, action))
            state = next_state
            reward_traj += reward
        print(f"Trajectory length: {len(traj)}, reward: {reward_traj}")
        trajectories.append(traj)
        reward_trajs.append(reward_traj)
    print(
        f"Average reward: {np.mean(reward_trajs)}, Avg traj length: {np.mean([len(traj) for traj in trajectories])}"
    )
    ipdb.set_trace()
    return trajectories


def get_pendulum_expert_traj_sac(env, num_traj):
    """
    Get expert trajectories for pendulum environment using the saved PPO checkpoint."""
    device = torch.device("cuda" if False else "cpu")
    policy = GaussianPolicy(
        env.observation_space.shape[0], env.action_space.shape[0], 256, env.action_space
    ).to(device)
    # checkpoint = torch.load("/home/sgurumur/locuslab/pytorch-soft-actor-critic/checkpoints/sac_checkpoint_Pendulum-v0_bestT200")
    checkpoint = torch.load("ckpts/sac/sac_checkpoint_Pendulum-v0_bestT200")
    policy.load_state_dict(checkpoint["policy_state_dict"])
    trajectories = []
    reward_trajs = []
    for _ in range(num_traj):
        state = env.reset()
        traj = []
        done = False
        reward_traj = 0
        while not done:
            action = (
                policy.sample(
                    torch.tensor(state, dtype=torch.float32).to(device)[None]
                )[2]
                .detach()
                .cpu()
                .numpy()[0]
            )
            next_state, reward, done, _ = env.step(action)
            traj.append((state, action))
            state = next_state
            reward_traj += reward
        print(f"Trajectory length: {len(traj)}, reward: {reward_traj}")
        trajectories.append(traj)
        reward_trajs.append(reward_traj)
    print(
        f"Average reward: {np.mean(reward_trajs)}, Avg traj length: {np.mean([len(traj) for traj in trajectories])}"
    )
    # ipdb.set_trace()
    return trajectories

def check_termination(state, num_goals, goal, env):
    """
    Check if the state is within the goal region.
    Args:
        state: The state of the environment.
        num_goals: The number of goals.
        goal: The goal state.
        env: The environment.
    Returns:
        A boolean value indicating if the state is within the goal region.
    """
    if torch.norm(state[0, :3] - goal[:3]) < 0.15 and (torch.norm(state[0, 6] - goal[6]) < 0.1):
        num_goals += 1
    else:
        num_goals = 0
    return num_goals

def check_termination2(state, num_goals, goal, env, done_traj):
    """
    Check if the state is within the goal region.
    Args:
        state: The state of the environment.
        num_goals: The number of goals.
        goal: The goal state.
        env: The environment.
    Returns:
        A boolean value indicating if the state is within the goal region.
    """
    # if torch.norm(state[0, :3] - goal[:3]) < 0.15 and ((state[0, 6] - goal[6]) < 0.1):
    #     num_goals += 1
    # else:
    #     num_goals = 0
    mask = torch.logical_and(torch.norm(state[0, 13] - goal[13]) < 0.1, ((state[:, 6] - goal[None,6]) < 0.1))
    num_goals = torch.where(done_traj, num_goals, torch.where(mask, num_goals + 1, torch.zeros_like(num_goals)))
    # if mask.float().mean() > 0:
    #     print(mask.mean())
    return num_goals*0

def get_expert_traj_cgac(env, num_traj):
    """
    Get expert trajectories for pendulum environment using the saved CGAC checkpoint."""
    device = torch.device("cpu")#cuda" if True else "cpu")
    policy = CGACGaussianPolicy(env.observation_space.shape[0], env.action_space.shape[0], [512,256], env.action_space, True).to(device)
    rms_obs = CGACRunningMeanStd((env.observation_space.shape[0],), device=device).to(device)
    # checkpoint = torch.load("/home/sgurumur/locuslab/pytorch-soft-actor-critic/checkpoints/sac_checkpoint_Pendulum-v0_bestT200")
    checkpoint = torch.load(f"ckpts/cgac/{env.saved_ckpt_name}")
    policy.load_state_dict(checkpoint["policy_state_dict"])
    rms_obs.load_state_dict(checkpoint["rms_obs"])
    trajectories = []
    reward_trajs = []
    while len(trajectories) < num_traj:
        state = env.reset()#.float()
        num_goals = 0
        state_rms = rms_obs(state.float())
        traj = []
        done = False
        reward_traj = 0
        while not done:
            action = policy.sample(state_rms)[2]
            next_state, reward, done, _ = env.step(action)
            num_goals = check_termination2(next_state, num_goals, env.targ_pos, env)
            traj.append((state[0].detach().cpu().numpy(), action[0].detach().cpu().numpy()))
            state = next_state#.float()
            state_rms = rms_obs(state.float())
            reward_traj += reward
            if num_goals > 5:
                break
        print(f"Trajectory length: {len(traj)}, reward: {reward_traj.item()}")
        # if reward_traj.item() < 100 and 'rexquadrotor' in env.spec_id:
        #     continue
        # if len(traj) < 150 and 'FlyingCartpole' in env.spec_id:
        #     continue
        trajectories.append(traj)
        reward_trajs.append(reward_traj.item())
        # ipdb.set_trace()
    print(
        f"Average reward: {np.mean(reward_trajs)}, Avg traj length: {np.mean([len(traj) for traj in trajectories])}"
    )
    # dynamics = lambda y: env.dynamics(y, torch.tensor(traj[0][1])[None])
    # print(env.dynamics(torch.tensor(trajectories[0][0][0])[None], torch.tensor(trajectories[0][0][1])[None]) - torch.tensor(trajectories[0][1][0])[None])
    # print(rk4(dynamics, torch.tensor(traj[0][0])[None], [0, env.dt]) - torch.tensor(traj[1][0])[None])
    ipdb.set_trace()
    return trajectories

def get_expert_traj_cgac2(env, num_traj):
    """
    Get expert trajectories for pendulum environment using the saved CGAC checkpoint."""
    device = torch.device("cpu")#cuda" if True else "cpu")
    policy = CGACGaussianPolicy(env.observation_space.shape[0], env.action_space.shape[0], [512,256], env.action_space, True).to(device)
    rms_obs = CGACRunningMeanStd((env.observation_space.shape[0],), device=device).to(device)
    # checkpoint = torch.load("/home/sgurumur/locuslab/pytorch-soft-actor-critic/checkpoints/sac_checkpoint_Pendulum-v0_bestT200")
    checkpoint = torch.load(f"ckpts/cgac/{env.saved_ckpt_name}")
    policy.load_state_dict(checkpoint["policy_state_dict"])
    rms_obs.load_state_dict(checkpoint["rms_obs"])
    trajectories = []
    reward_trajs = []
    while len(trajectories) < num_traj:
        state = env.reset()#.float()
        bsz = env.bsz
        num_goals = torch.zeros(bsz)
        state_rms = rms_obs(state.float())
        traj = [[] for _ in range(bsz)]
        states = []
        done = False
        reward_traj = torch.zeros(bsz)
        done_traj = torch.zeros(bsz, dtype=torch.bool)
        while not done_traj.all():
            action = policy.sample(state_rms)[2]
            action = env.action_clip(action)
            # ipdb.set_trace()
            next_state, reward, done, _ = env.step(action)
            num_goals = check_termination2(next_state, num_goals, env.targ_pos, env, done_traj)
            for i in range(bsz):
                if done_traj[i]:
                    continue
                traj[i].append((state[i].detach().cpu().numpy(), action[i].detach().cpu().numpy()))
                reward_traj[i] += reward[i]
            states.append(state.detach().cpu().numpy())
            state = next_state#.float()
            state_rms = rms_obs(state.float())
            
            done_traj = torch.logical_or(torch.logical_or(done_traj, num_goals > 10), done)
            # if (num_goals > 10).all():
            #     break
        # states = np.stack(states, axis=1)
        # ipdb.set_trace()
        print(f"Trajectory length: {np.mean([len(traj[i]) for i in range(bsz)])}, reward: {reward_traj.mean().item()}")
        if 'rexquadrotor' in env.spec_id:
            mask = reward_traj.item() >= 100
        if 'FlyingCartpole' in env.spec_id:
            mask = num_goals >= -10 #and len(traj) < 150
        for i in range(bsz):
            if mask[i]:
                trajectories.append(traj[i])
                reward_trajs.append(reward_traj[i].item())
    print(
        f"Average reward: {np.mean(reward_trajs)}, Avg traj length: {np.mean([len(traj) for traj in trajectories])}"
    )
    # dynamics = lambda y: env.dynamics(y, torch.tensor(traj[0][1])[None])
    # print(env.dynamics(torch.tensor(trajectories[0][0][0])[None], torch.tensor(trajectories[0][0][1])[None]) - torch.tensor(trajectories[0][1][0])[None])
    # print(rk4(dynamics, torch.tensor(traj[0][0])[None], [0, env.dt]) - torch.tensor(traj[1][0])[None])
    ipdb.set_trace()
    return trajectories

def get_expert_traj_cgac_mpc(env, num_traj):
    """
    Get expert trajectories for pendulum environment using the saved CGAC checkpoint."""
    device = torch.device("cpu")#cuda" if True else "cpu")
    policy = CGACGaussianPolicy(env.observation_space.shape[0], env.action_space.shape[0], [512,256], env.action_space, True).to(device)
    rms_obs = CGACRunningMeanStd((env.observation_space.shape[0],), device=device).to(device)

    parser = argparse.ArgumentParser()
    parser.add_argument("--nq", type=int, default=7)  # observation (configurations) for the policy
    parser.add_argument("--T", type=int, default=20)  # look-ahead horizon length (including current time step)
    parser.add_argument("--H", type=int, default=1)  # observation history length (including current time step)
    parser.add_argument("--qp_iter", type=int, default=2)
    parser.add_argument("--eps", type=float, default=1e-2)
    parser.add_argument("--warm_start", type=bool, default=True)
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--solver_type", type=str, default='al')
    parser.add_argument("--dtype", type=str, default="double")
    args = parser.parse_args()
    args.env = env
    args.Q = env.Qlqr.to(args.device)
    args.Qaux = env.Qaux.to(args.device)
    args.R = env.Rlqr.to(args.device)

    mpc_policy = Tracking_MPC(args, env).to(device)
    # checkpoint = torch.load("/home/sgurumur/locuslab/pytorch-soft-actor-critic/checkpoints/sac_checkpoint_Pendulum-v0_bestT200")
    checkpoint = torch.load(f"ckpts/cgac/{env.saved_ckpt_name}")
    policy.load_state_dict(checkpoint["policy_state_dict"])
    rms_obs.load_state_dict(checkpoint["rms_obs"])
    trajectories = []
    reward_trajs = []
    goal = env.targ_pos
    while len(trajectories) < num_traj:
        state = env.reset()#.float()
        num_goals = 0
        state_rms = rms_obs(state.float())
        traj = []
        done = False
        reward_traj = 0
        while not done:
            switch = (torch.norm(state[0, 6] - goal[6]) < 0.15) and (torch.norm(state[0, 13] - goal[13]) < 0.25)
            if switch:   
                print("switching to MPC...")
                x_ref = goal.repeat(args.T, 1)
                mpc_policy.reinitialize(x_ref, torch.ones(args.bsz, args.T, 1))
                u_ref = torch.zeros(args.T, env.nu)
                xu_ref = torch.cat([x_ref, u_ref], dim=-1)
                _, nominal_actions = mpc_policy(state, xu_ref, x_ref, u_ref, al_iters=10)
                action = nominal_actions[0][0].unsqueeze(0)
                # ipdb.set_trace()       
            else:
                action = policy.sample(state_rms)[2]
            next_state, reward, done, _ = env.step(action)
            num_goals = check_termination(next_state, num_goals, goal, env)
            traj.append((state[0].detach().cpu().numpy(), action[0].detach().cpu().numpy()))
            state = next_state#.float()
            state_rms = rms_obs(state.float())
            reward_traj += reward
            if num_goals > 5:
                break
        print(f"Trajectory length: {len(traj)}, reward: {reward_traj.item()}")
        # if reward_traj.item() < 100 and 'rexquadrotor' in env.spec_id:
        #     continue
        # if len(traj) < 150 and 'FlyingCartpole' in env.spec_id:
        #     continue
        if len(traj) >= 199:
            print("not this traj")
            continue
        trajectories.append(traj)
        reward_trajs.append(reward_traj.item())
        # ipdb.set_trace()
    print(
        f"Average reward: {np.mean(reward_trajs)}, Avg traj length: {np.mean([len(traj) for traj in trajectories])}"
    )
    # dynamics = lambda y: env.dynamics(y, torch.tensor(traj[0][1])[None])
    # print(env.dynamics(torch.tensor(trajectories[0][0][0])[None], torch.tensor(trajectories[0][0][1])[None]) - torch.tensor(trajectories[0][1][0])[None])
    # print(rk4(dynamics, torch.tensor(traj[0][0])[None], [0, env.dt]) - torch.tensor(traj[1][0])[None])
    ipdb.set_trace()
    return trajectories

def save_expert_traj(env, num_traj, type="mpc"):
    """Save expert trajectories to a file.

    Args:
        env: gym environment
        num_traj: number of trajectories to save
        type: type of expert to use
    """

    ## use env name to choose which function to use to get expert trajectories
    if env.spec_id == "Pendulum-v0" or env.spec_id == "Pendulum-v0-stabilize":
        if type == "mpc":
            expert_traj = get_pendulum_expert_traj_mpc(env, num_traj)
        elif type == "ppo":
            expert_traj = get_pendulum_expert_traj_ppo(env, num_traj)
        elif type == "sac":
            expert_traj = get_pendulum_expert_traj_sac(env, num_traj)
    elif env.spec_id == "Integrator-v0":
        if type == "mpc":
            expert_traj = get_int_expert_traj_mpc(env, num_traj)
        else:
            raise NotImplementedError
    elif type == "cgac":
        expert_traj = get_expert_traj_cgac2(env, num_traj)

    ## save expert trajectories to a file in data folder
    if os.path.exists("data") == False:
        os.makedirs("data")

    with open(f"data/expert_traj_{type}-{env.spec_id}-ub0.3-L0.50-swing.pkl", "wb") as f:
        pickle.dump(expert_traj, f)


def get_gt_data(args, env, type="mpc"):
    """
    Get ground truth data for imitation learning.
    Args:
        args: The arguments for the training script.
        env: The environment.
        type: The type of controller to use. Can be 'mpc' or 'ppo' or 'sac'.
    Returns:
        A list of trajectories, each trajectory is a list of (state, action) tuples.
    """
    # with open('data/expert_traj_mpc-Pendulum-v0.pkl', 'rb') as f:#f'data/expert_traj_{type}-{env.spec_id}.pkl', 'rb') as f:
    # with open(f"data/expert_traj_{type}-{env.spec_id}-ub0.3-swing-clip_new.pkl", "rb") as f:
    with open(f"data/expert_traj_{type}-{env.spec_id}-ub03-clip-s_new.pkl", "rb") as f:
        gt_trajs = pickle.load(f)
    # states = [[] for i in range(len(gt_trajs))]
    # for j in range(len(gt_trajs)):
    #     for i in range(len(gt_trajs[0])):
    #         states[j].append(gt_trajs[j][i][0])

    # ipdb.set_trace()
    return gt_trajs


def merge_gt_data(gt_trajs, num_trajs=2):
    """
    Merge ground truth data for imitation learning.
    Args:
        gt_trajs: A list of trajectories, each trajectory is a list of (state, action) tuples.
    Returns:
        A list of (state, action) tuples.
    """
    merged_gt_traj = {"state": [], "action": [], "mask": []}
    for i, traj in enumerate(gt_trajs):
        if i >= num_trajs:
            break
        for state, action in traj:
            merged_gt_traj["state"].append(state)
            merged_gt_traj["action"].append(action)
            merged_gt_traj["mask"].append(1)
        merged_gt_traj["mask"][-1] = 0  # mask = 0 at the end of each trajectory
    # ipdb.set_trace()
    merged_gt_traj["state"] = torch.tensor(
        np.array(merged_gt_traj["state"]), dtype=torch.float32
    )
    merged_gt_traj["action"] = torch.tensor(
        np.array(merged_gt_traj["action"]), dtype=torch.float32
    )
    merged_gt_traj["mask"] = torch.tensor(
        np.array(merged_gt_traj["mask"]), dtype=torch.float32
    )
    return merged_gt_traj

def sample_trajectory(gt_trajs, bsz, H, T):
    """
    Sample a batch of trajectories from the ground truth data.
    Args:
        gt_trajs: A dictionary of "state", "action" and "mask" tensors with concatenated trajectories.
        bsz: Batch size.
        H: History length.
        T: Lookahead horizon length.
    Returns:
        A list of trajectories, each trajectory is a list of (obs, state, action) tuples (H, T, T).
    """
    idxs = np.random.randint(H-1, len(gt_trajs["state"]), bsz*4)
    trajs = {"obs": [], "obs_action": [], "state": [], "action": [], "mask": []}
    i = 0
    j = 0
    while j < bsz:
        if (gt_trajs["mask"][idxs[i]+1-H:idxs[i]+1] == 0).sum()>0:
            i += 1
            continue
        trajs["obs"].append(gt_trajs["state"][idxs[i]+1 - H : idxs[i]+1])
        trajs["obs_action"].append(gt_trajs["action"][idxs[i]+1 - H : idxs[i]+1])
        if idxs[i] + T <= len(gt_trajs["state"]):
            trajs["state"].append(gt_trajs["state"][idxs[i] : idxs[i] + T])
            trajs["action"].append(gt_trajs["action"][idxs[i] : idxs[i] + T])
            trajs["mask"].append(gt_trajs["mask"][idxs[i] : idxs[i] + T])
        else:
            padding = idxs[i] + T - len(gt_trajs["state"])            
            trajs["state"].append(
                torch.cat(
                    [gt_trajs["state"][idxs[i] :], gt_trajs["state"][:padding] * 0.0],
                    dim=0,
                )
            )
            trajs["action"].append(
                torch.cat(
                    [gt_trajs["action"][idxs[i] :], gt_trajs["action"][:padding] * 0.0],
                    dim=0,
                )
            )
            trajs["mask"].append(
                torch.cat(
                    [gt_trajs["mask"][idxs[i] :], gt_trajs["mask"][:padding] * 0.0], dim=0
                )
            )
        i += 1
        j += 1
    trajs["obs"] = torch.stack(trajs["obs"])
    trajs["state"] = torch.stack(trajs["state"])
    trajs["action"] = torch.stack(trajs["action"])
    trajs["mask"] = torch.stack(trajs["mask"])
    trajs["obs_action"] = torch.stack(trajs["obs_action"])
    for i in reversed(range(T)):
        trajs["mask"][:, i] = torch.prod(trajs["mask"][:, :i+1], dim=1)
    # trajs["state"] = trajs["state"]*trajs["mask"][:, :, None]
    # trajs["action"] = trajs["action"]*trajs["mask"][:, :, None]
    return trajs

def test_qp_mpc(env):
    mpc_controller = PendulumExpert(env)
    trajectories = []
    state = env.reset()  # Reset environment to a new initial state
    traj = []
    done = False
    actions, states = mpc_controller.optimize_action(
        torch.tensor(state, dtype=torch.float32).view(1, -1)
    )
    # ipdb.set_trace()
    states = states.squeeze().detach().numpy()
    actions = actions.detach().numpy()
    traj = []
    trajectories = []
    ipdb.set_trace()
    for i in range(len(states)):
        traj.append((states[i], actions[i][0]))
    trajectories.append(traj)
    with open(f"data/expert_traj_mpc-{env.spec_id}.pkl", "wb") as f:
        pickle.dump(trajectories, f)


class IntegratorExpert:
    def __init__(self, env, type="mpc"):
        """
        Initialize the MPC controller with the necessary parameters.

        Args:
            env: The PendulumEnv environment.
            type: The type of controller to use. Can be 'mpc' or 'ppo' or 'sac'.
        """

        self.type = type

        if self.type == "mpc":
            self.T = 20
            self.goal_state = torch.Tensor([0.0, 0.0])
            self.goal_weights = torch.Tensor([10.0, 1])
            self.ctrl_penalty = 0.001
            self.mpc_eps = 1e-3
            self.linesearch_decay = 0.2
            self.max_linesearch_iter = 5
            self.nx = env.observation_space.shape[0]
            self.nu = env.action_space.shape[0]
            self.bsz = 1

            self.u_lower = torch.tensor(
                env.action_space.low, dtype=torch.float32
            ).double()
            self.u_upper = torch.tensor(
                env.action_space.high, dtype=torch.float32
            ).double()

            self.qp_iter = 1
            self.u_init = torch.zeros(self.T, self.bsz, self.nu).double()
            self.q = torch.cat(
                (self.goal_weights, self.ctrl_penalty * torch.ones(self.nu))
            )
            self.px = -torch.sqrt(self.goal_weights) * self.goal_state
            self.p = torch.cat((self.px, torch.zeros(self.nu)))
            self.Q = (
                torch.diag(self.q)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(self.T, self.bsz, 1, 1)
                .double()
            )
            self.p = self.p.unsqueeze(0).repeat(self.T, self.bsz, 1).double()

            self.ctrl = mpc.MPC(
                self.nx,
                self.nu,
                self.T,
                u_lower=self.u_lower,
                u_upper=self.u_upper,  # .double(),
                qp_iter=self.qp_iter,
                exit_unconverged=False,
                eps=1e-2,
                n_batch=self.bsz,
                backprop=False,
                verbose=0,
                u_init=self.u_init,  # .double(),
                grad_method=mpc.GradMethods.AUTO_DIFF,
                solver_type="dense",
                single_qp_solve=True,  # linear system
            )
            self.cost = mpc.QuadCost(self.Q, self.p)

    def optimize_action(self, state):
        """Solve the MPC problem for the given state."""
        # ipdb.set_trace()
        nominal_states, nominal_actions = self.ctrl(
            state.double(), self.cost, env.dynamics
        )
        u = torch.clamp(nominal_actions[0], self.u_lower, self.u_upper)
        return u  # Return the first action in the optimal sequence


def get_int_expert_traj_mpc(env, num_traj):
    """
    Get expert trajectories for integrator environment using MPC for trajectory optimization.
    Args:
        env: The IntegratorEnv environment.
        mpc_controller: The IntegratorExpert.
        num_traj: Number of trajectories to save.
    Returns:
        A list of trajectories, each trajectory is a list of (state, action) tuples.
    """
    mpc_controller = IntegratorExpert(env)
    trajectories = []
    for _ in range(num_traj):
        state = env.reset()  # Reset environment to a new initial state
        traj = []
        done = False
        while not done:
            action = mpc_controller.optimize_action(
                torch.tensor(state, dtype=torch.float32).view(1, -1)
            )
            # ipdb.set_trace()
            next_state, _, done, _ = env.step(action)
            traj.append((state, action.numpy()[0]))
            state = next_state
            # ipdb.set_trace()
            if len(traj) > 100:
                ipdb.set_trace()
            print(len(traj))
        print(f"Trajectory length: {len(traj)}")
        trajectories.append(traj)
    return trajectories

def seeding(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == "__main__":
    seeding(2)
    print("Starting!")
    device = "cuda:0"
    kwargs = {"dtype": torch.float64, "device": device, "requires_grad": False}
    # ipdb.set_trace()
    # env = PendulumEnv(stabilization=False)
    # env = RexQuadrotor(bsz=1)
    # env = CartpoleEnv(nx=4, dt=0.05, stabilization=False, kwargs=kwargs)
    # ipdb.set_trace()
    # env = Cartpole2linkEnv(dt=0.05, stabilization=False, kwargs=kwargs)
    env = FlyingCartpole(bsz=100, max_steps=200)
    # env = IntegratorEnv()
    # save_expert_traj(env, 300, "sac")
    # save_expert_traj(env, 2, "mpc")
    save_expert_traj(env, 100, "cgac")
    # test_qp_mpc(env)
