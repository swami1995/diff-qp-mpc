import math
import time

import numpy as np
import torch
import torch.autograd as autograd
import qpth.qp_wrapper as mpc
import ipdb
import os
from envs import PendulumEnv, PendulumDynamics
from ppo_train import PPO, GaussianPolicy
import pickle

class MPCPendulumController:
    def __init__(self, env):
        """
        Initialize the MPC controller with the necessary parameters.
        Args:
            nx (int): Number of state dimensions.
            nu (int): Number of control dimensions.
            T (int): Time horizon for the MPC.
            u_lower (float): Lower bound for control signal.
            u_upper (float): Upper bound for control signal.
            max_iter (int): Maximum iterations for the QP solver.
            bsz (int): Batch size for MPC optimization.
            u_init (float): Initial guess for control signal.
            Q (np.ndarray): State cost matrix.
            p (np.ndarray): Control cost matrix.
        """
        nx = env.observation_space.shape[0]
        nu = env.action_space.shape[0]
        T = env.T
        u_lower = torch.tensor(env.action_space[:, 0], dtype=torch.float32)
        u_upper = torch.tensor(env.action_space[:, 1], dtype=torch.float32)
        max_iter = 1
        bsz = 1
        u_init = torch.zeros(T, bsz, nu)
        q,p = env.get_true_obj()
        Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(
            T, bsz, 1, 1
        )
        p = p.unsqueeze(0).repeat(T, bsz, 1)
        self.ctrl = mpc.MPC(
            nx, nu, T, 
            u_lower=u_lower, u_upper=u_upper, 
            qp_iter=max_iter, exit_unconverged=False, 
            eps=1e-2, n_batch=bsz, backprop=False, 
            verbose=0, u_init=u_init, 
            grad_method=mpc.GradMethods.AUTO_DIFF, 
            solver_type='dense'
        )
        self.cost = mpc.QuadCost(Q, p)

    def optimize_action(self, state):
        """Solve the MPC problem for the given state."""
        # ipdb.set_trace()
        nominal_states, nominal_actions = self.ctrl(state, self.cost, PendulumDynamics())
        return nominal_actions[0]  # Return the first action in the optimal sequence

def get_pendulum_expert_traj_mpc(env, num_traj):
    """
    Get expert trajectories for pendulum environment using MPC for trajectory optimization.
    Args:
        env: The PendulumEnv environment.
        mpc_controller: The MPCPendulumController.
        num_traj: Number of trajectories to save.
    Returns:
        A list of trajectories, each trajectory is a list of (state, action) tuples.
    """
    mpc_controller = MPCPendulumController(env)
    trajectories = []
    for _ in range(num_traj):
        state = env.reset()  # Reset environment to a new initial state
        traj = []
        done = False
        while not done:
            action = mpc_controller.optimize_action(torch.tensor(state, dtype=torch.float32).view(1, -1))
            next_state, _, done, _ = env.step(action)
            traj.append((state, action.numpy()))
            state = next_state
            # if len(traj) >= env.T:
            #     ipdb.set_trace()
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
    ppo_policy = PPO(state_dim, action_dim, lr, lr, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
    ppo_policy.load('PPO_preTrained/Pendulum-v0/PPO_Pendulum-v0_0_0.pth')
    trajectories = []
    reward_trajs = []
    for _ in range(num_traj):
        state = env.reset()
        traj = []
        done = False
        reward_traj = 0
        while not done:
            action = ppo_policy.policy.actor(torch.tensor(state, dtype=torch.float32).to(ppo_policy.policy.action_var.device)).detach().cpu().numpy()
            next_state, reward, done, _ = env.step(action)
            traj.append((state, action))
            state = next_state
            reward_traj += reward
        print(f"Trajectory length: {len(traj)}, reward: {reward_traj}")
        trajectories.append(traj)
        reward_trajs.append(reward_traj)
    print(f"Average reward: {np.mean(reward_trajs)}, Avg traj length: {np.mean([len(traj) for traj in trajectories])}")
    ipdb.set_trace()
    return trajectories

def get_pendulum_expert_traj_sac(env, num_traj):
    """
    Get expert trajectories for pendulum environment using the saved PPO checkpoint."""
    device = torch.device("cuda" if True else "cpu")
    policy = GaussianPolicy(env.observation_space.shape[0], env.action_space.shape[0], 256, env.action_space).to(device)
    checkpoint = torch.load("/home/sgurumur/locuslab/pytorch-soft-actor-critic/checkpoints/sac_checkpoint_Pendulum-v0_bestT200")
    policy.load_state_dict(checkpoint['policy_state_dict'])
    trajectories = []
    reward_trajs = []
    for _ in range(num_traj):
        state = env.reset()
        traj = []
        done = False
        reward_traj = 0
        while not done:
            action = policy.sample(torch.tensor(state, dtype=torch.float32).to(device)[None])[2].detach().cpu().numpy()[0]
            next_state, reward, done, _ = env.step(action)
            traj.append((state, action))
            state = next_state
            reward_traj += reward
        print(f"Trajectory length: {len(traj)}, reward: {reward_traj}")
        trajectories.append(traj)
        reward_trajs.append(reward_traj)
    print(f"Average reward: {np.mean(reward_trajs)}, Avg traj length: {np.mean([len(traj) for traj in trajectories])}")
    ipdb.set_trace()
    return trajectories


def save_expert_traj_mpc(env, num_traj):
    """Save expert trajectories to a file.

    Args:
        env: gym environment
        num_traj: number of trajectories to save
    """

    ## use env name to choose which function to use to get expert trajectories
    if env.spec_id == 'Pendulum-v0':
        expert_traj = get_pendulum_expert_traj_mpc(env, num_traj)
    else:
        raise NotImplementedError
    
    ## save expert trajectories to a file in data folder
    if os.path.exists('data') == False:
        os.makedirs('data')
        
    with open(f'data/expert_traj-{env.spec_id}.pkl', 'wb') as f:
        pickle.dump(expert_traj, f)

def save_expert_traj_ppo(env, num_traj):
    """Save expert trajectories to a file.

    Args:
        env: gym environment
        num_traj: number of trajectories to save
    """

    ## use env name to choose which function to use to get expert trajectories
    if env.spec_id == 'Pendulum-v0':
        expert_traj = get_pendulum_expert_traj_ppo(env, num_traj)
    else:
        raise NotImplementedError
    
    ## save expert trajectories to a file in data folder
    if os.path.exists('data') == False:
        os.makedirs('data')
    
    with open(f'data/expert_traj_ppo-{env.spec_id}.pkl', 'wb') as f:
        pickle.dump(expert_traj, f)

def save_expert_traj_sac(env, num_traj):
    """Save expert trajectories to a file.

    Args:
        env: gym environment
        num_traj: number of trajectories to save
    """

    ## use env name to choose which function to use to get expert trajectories
    if env.spec_id == 'Pendulum-v0':
        expert_traj = get_pendulum_expert_traj_sac(env, num_traj)
    elif env.spec_id == 'Pendulum-v0-stabilize':
        expert_traj = get_pendulum_expert_traj_sac(env, num_traj)
    else:
        raise NotImplementedError
    
    ## save expert trajectories to a file in data folder
    if os.path.exists('data') == False:
        os.makedirs('data')
    
    with open(f'data/expert_traj_sac-{env.spec_id}.pkl', 'wb') as f:
        pickle.dump(expert_traj, f)

def get_gt_data(args, env):
    """
    Get ground truth data for imitation learning.
    Args:
        args: The arguments for the training script.
        env: The environment.
    Returns:
        A list of trajectories, each trajectory is a list of (state, action) tuples.
    """
    with open(f'data/expert_traj_sac-{env.spec_id}.pkl', 'rb') as f:
        gt_trajs = pickle.load(f)
    return gt_trajs

def merge_gt_data(gt_trajs):
    """
    Merge ground truth data for imitation learning.
    Args:
        gt_trajs: A list of trajectories, each trajectory is a list of (state, action) tuples.
    Returns:
        A list of (state, action) tuples.
    """
    merged_gt_traj = {"state": [], "action": [], "mask": []}
    for traj in gt_trajs:
        for state, action in traj:
            merged_gt_traj["state"].append(state)
            merged_gt_traj["action"].append(action)
            merged_gt_traj["mask"].append(1)
        merged_gt_traj["mask"][-1] = 0
    merged_gt_traj["state"] = torch.tensor(np.array(merged_gt_traj["state"]), dtype=torch.float32)
    merged_gt_traj["action"] = torch.tensor(np.array(merged_gt_traj["action"]), dtype=torch.float32)
    merged_gt_traj["mask"] = torch.tensor(np.array(merged_gt_traj["mask"]), dtype=torch.float32)
    return merged_gt_traj

def sample_trajectory(gt_trajs, bsz, T):
    """
    Sample a batch of trajectories from the ground truth data.
    Args:
        gt_trajs: A dictionary of "state", "action" and "mask" tensors with concatenated trajectories.
        bsz: Batch size.
        T: Time horizon.
    Returns:
        A list of trajectories, each trajectory is a list of (state, action) tuples with length T.
    """
    idxs = np.random.randint(0, len(gt_trajs["state"]), bsz)
    trajs = {"state": [], "action": [], "mask": []}
    for i in range(bsz):
        if idxs[i] + T < len(gt_trajs["state"]):
            trajs["state"].append(gt_trajs["state"][idxs[i]:idxs[i]+T])
            trajs["action"].append(gt_trajs["action"][idxs[i]:idxs[i]+T])
            trajs["mask"].append(gt_trajs["mask"][idxs[i]:idxs[i]+T])
        else:
            padding = idxs[i] + T - len(gt_trajs["state"])
            trajs["state"].append(torch.cat([gt_trajs["state"][idxs[i]:], gt_trajs["state"][:padding]*0.0], dim=0))
            trajs["action"].append(torch.cat([gt_trajs["action"][idxs[i]:], gt_trajs["action"][:padding]*0.0], dim=0))
            trajs["mask"].append(torch.cat([gt_trajs["mask"][idxs[i]:], gt_trajs["mask"][:padding]*0], dim=0))
    trajs["state"] = torch.stack(trajs["state"])
    trajs["action"] = torch.stack(trajs["action"])
    trajs["mask"] = torch.stack(trajs["mask"])
    for i in range(T):
        trajs["mask"][:, i] = torch.prod(trajs["mask"][:, :i], dim=1)
    return trajs

if __name__ == '__main__':
    print("Starting!")
    # ipdb.set_trace()
    env = PendulumEnv(stabilization=True)
    save_expert_traj_sac(env, 200)