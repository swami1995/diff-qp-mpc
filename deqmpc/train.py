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
from deq_layer_utils import update_scales

# torch.set_default_device('cuda')
np.set_printoptions(precision=4, suppress=True)
# import tensorboard from pytorch

# example task : hard pendulum with weird coordinates to make sure direct target tracking is difficult


def seeding(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="pendulum")
    parser.add_argument("--nq", type=int, default=1)  # observation (configurations) for the policy
    parser.add_argument("--T", type=int, default=5)  # look-ahead horizon length (including current time step)
    parser.add_argument("--H", type=int, default=1)  # observation history length (including current time step)
    # parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument("--qp_iter", type=int, default=1)
    parser.add_argument("--eps", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warm_start", type=bool, default=True)
    parser.add_argument("--bsz", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--deq", action="store_true")
    parser.add_argument("--hdim", type=int, default=128)
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
    parser.add_argument("--pooling", type=str, default="sum")
    parser.add_argument("--solver_type", type=str, default='al')
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--dtype", type=str, default="double")
    parser.add_argument("--ckpt", type=str, default="bc_sac_pen")
    parser.add_argument("--deq_out_type", type=int, default=1)  # previously 1
    parser.add_argument("--policy_out_type", type=int, default=1)  # previously 1
    parser.add_argument("--loss_type", type=str, default='l1')  # previously 0
    parser.add_argument("--deq_reg", type=float, default=0.1) # previously 0.0
    # check noise_utils.py for noise_type
    parser.add_argument("--data_noise_type", type=int, default=0)
    parser.add_argument("--data_noise_std", type=float, default=0.05)
    parser.add_argument("--data_noise_mean", type=float, default=0.3)
    parser.add_argument("--grad_coeff", action="store_true")
    parser.add_argument("--scaled_output", action="store_true")

    args = parser.parse_args()
    seeding(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device if args.device is None else args.device
    if args.save:
        if (args.qp_solve):
            method_name = f"deqmpc_" 
        elif (args.lastqp_solve):
            method_name = f"diffmpc_"
        args.name = method_name + args.name + \
            f"_T{args.T}_bsz{args.bsz}_deq_iter{args.deq_iter}_hdim{args.hdim}"
        writer = SummaryWriter("./logs/" + args.name)
        print("logging to: ", args.name)

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

    # gt_trajs = get_gt_data(args, env, "mpc")
    # gt_trajs = get_gt_data(args, env, "cgac")
    # ipdb.set_trace()
    gt_trajs = merge_gt_data(gt_trajs)
    args.Q = env.Qlqr.to(args.device)
    args.R = env.Rlqr.to(args.device)

    traj_sample = sample_trajectory(gt_trajs, args.bsz*10, args.H, args.T)
    if args.env == "pendulum":
        traj_sample["state"] = utils.unnormalize_states_pendulum(
            traj_sample["state"])
    elif args.env == "cartpole1link" or args.env == "cartpole2link":
        traj_sample["state"] = utils.unnormalize_states_cartpole_nlink(
            traj_sample["state"])
    args.max_scale = ((traj_sample["state"] - traj_sample["state"][:, :1])*traj_sample["mask"][:, :, None]).reshape(args.bsz*50,4).abs().max(dim=0)[0].to(args.device)
    if args.deq:
        policy = DEQMPCPolicy(args, env).to(args.device)
        # policy = DEQMPCPolicyHistory(args, env).to(args.device)
        # policy = DEQMPCPolicyFeedback(args, env).to(args.device)
        # policy = DEQMPCPolicyQ(args, env).to(args.device)
        # save arguments
        if args.save:
            torch.save(args, "./logs/" + args.name + "/args")
    else:
        # policy = NNMPCPolicy(args, env).to(args.device)
        policy = NNPolicy(args, env).to(args.device)
        # save arguments
        torch.save(args, "./model/bc_sac_pen_args")

    if args.load:
        policy.load_state_dict(torch.load(f"./model/{args.ckpt}"))
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=args.lr)
    losses = []
    losses_end = []
    time_diffs = []
    dyn_resids = []
    losses_var = []
    losses_iter = [[] for _ in range(args.deq_iter)]
    if args.deq_out_type == 0:
        num_coeffs_per_iter = 1
    elif args.deq_out_type == 1:
        num_coeffs_per_iter = 2
    elif args.deq_out_type == 2:
        num_coeffs_per_iter = 3
    elif args.deq_out_type == 3:
        num_coeffs_per_iter = 1
    coeffs = torch.ones((args.deq_iter, num_coeffs_per_iter), device=args.device)
    # coeffs = torch.tensor([[1.0, 6.0, 8.0, 10.0, 12.0, 14.0], 
    #                        [0.03, 0.06, 0.08, 0.10, 0.12, 0.14]],
    #                        device=args.device).t()
    # coeffs = torch.tensor([[1.0, 6.0, 7.0, 8.0, 9.0, 10.0], 
    #                        [0.03, 0.06, 0.07, 0.8, 0.9, 0.1]],
    #                        device=args.device).t()

    losses_iter_opt = [[] for _ in range(args.deq_iter)]
    losses_iter_nn = [[] for _ in range(args.deq_iter)]
    losses_iter_base = [[] for _ in range(args.deq_iter)]

    # run imitation learning using gt_trajs
    for i in range(20000):
        # sample bsz random trajectories from gt_trajs and a random time step for each
        traj_sample = sample_trajectory(gt_trajs, args.bsz, args.H, args.T)
        traj_sample = {k: v.to(args.device) for k, v in traj_sample.items()}
        # ipdb.set_trace()

        if args.env == "pendulum":
            traj_sample["state"] = utils.unnormalize_states_pendulum(
                traj_sample["state"])
            traj_sample["obs"] = utils.unnormalize_states_pendulum(traj_sample["obs"])
        elif args.env == "cartpole1link" or args.env == "cartpole2link":
            traj_sample["state"] = utils.unnormalize_states_cartpole_nlink(
                traj_sample["state"])
            traj_sample["obs"] = utils.unnormalize_states_pendulum(traj_sample["obs"])
        pretrain_done = False if (i < 5000 and args.pretrain) else True
        # warm start only after 1000 iterations

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
        # ipdb.set_trace()
        if args.deq:
            start = time.time()
            policy_out = policy(obs_in, gt_states, gt_actions,
                                gt_mask, out_iter=i, qp_solve=args.qp_solve and pretrain_done, lastqp_solve=args.lastqp_solve and pretrain_done)
            end = time.time()
            dyn_resids.append(policy_out["dyn_res"])
        else:
            policy_out = policy(obs_in)
        # if (i % 2000 == 0):
            # ipdb.set_trace()
        if (i % 20 == 0):
            coeffs_est, losses_nocoeff, losses_proxy_nocoeff = policies.compute_grad_coeff(policy, gt_states, gt_actions, gt_mask, policy_out["trajs"], args.deq, pretrain_done)
            coeffs_est = coeffs_est.view(args.deq_iter, num_coeffs_per_iter)
            if args.grad_coeff:
                coeffs = coeffs_est*0.2 + coeffs*0.8
            [losses_iter_nocoeff[k].append(losses_nocoeff[k].item()) for k in range(args.deq_iter)]
            [losses_proxy_iter_nocoeff[k].append(losses_proxy_nocoeff[k].item()) for k in range(args.deq_iter)]
        
        if args.scaled_output:
            update_scales(policy, policy_out["trajs"], gt_states, policy_out["init_states"], gamma=0.95)

        # if (i % 1000 == 0):
        #     ipdb.set_trace()

        # if (i == 5500):
        #     ipdb.set_trace()

        loss_dict = policies.compute_loss(policy, gt_states, gt_actions, gt_mask, policy_out, args.deq, pretrain_done, coeffs)
        loss = loss_dict["loss"]
        time_diffs.append(end-start)
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        # gradient clipping
        # torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 4)
        # ipdb.set_trace()
        optimizer.step()

        loss_end = loss_dict["loss_end"]
        losses_end.append(loss_end.item())
        losses_var.append(loss_dict["losses_var"])
        [losses_iter[k].append(loss_dict["losses_iter"][k]) for k in range(args.deq_iter)]
        [losses_iter_opt[k].append(loss_dict["losses_iter_opt"][k]) for k in range(args.deq_iter)]
        [losses_iter_nn[k].append(loss_dict["losses_iter_nn"][k]) for k in range(args.deq_iter)]
        [losses_iter_base[k].append(loss_dict["losses_iter_base"][k]) for k in range(args.deq_iter)]

        # Printing
        if i % 100 == 0:
            print("iter: ", i, "deqmpc" if pretrain_done else "deq")
            print(
                "grad norm: ",
                torch.nn.utils.clip_grad_norm_(
                    policy.model.parameters(), 1000),
            )
            print(
                "loss: ",
                np.mean(losses) / args.deq_iter,
                "loss_end: ",
                np.mean(losses_end),
                "avg time: ",
                np.mean(time_diffs),
                "dyn res: ",
                np.mean(dyn_resids),
            )
            if args.save:
                torch.save(policy.state_dict(), "./model/" + args.name)
                writer.add_scalar("losses/loss_avg",
                                  np.mean(losses) / args.deq_iter, i)
                writer.add_scalar("losses/loss_end", np.mean(losses_end), i)
                
                [writer.add_scalar(f"losses/loss{k}", np.mean(losses_iter[k]), i) for k in range(len(losses_iter))]
                [writer.add_scalar(f"coeffs/coeff{j}{k}", coeffs[j,k], i) for j in range(args.deq_iter) for k in range(num_coeffs_per_iter)]
                [writer.add_scalar(f"coeffs_res/coeff{k}", loss_dict["iter_weights"].mean(dim=0)[k], i) for k in range(args.deq_iter)]
                writer.add_scalar("coeffs_res/coeff_ex_var", loss_dict["ex_weights"].var(), i)
                [writer.add_scalar(f"losses_nocoeff/loss_nocoeff{k}", np.mean(losses_iter_nocoeff[k]), i) for k in range(len(losses_iter_nocoeff))]
                [writer.add_scalar(f"losses_nocoeff/loss_proxy_nocoeff{k}", np.mean(losses_proxy_iter_nocoeff[k]), i) for k in range(len(losses_proxy_iter_nocoeff))]
                # [writer.add_scalar(f"scales/scale{k}", policy.model.scales[k][0,0], i) for k in range(len(scales))]
                [writer.add_scalar(f"losses_opt/losses_iter_opt{k}", np.mean(losses_iter_opt[k]), i) for k in range(len(losses_iter_opt))]
                [writer.add_scalar(f"losses_nn/losses_iter_nn{k}", np.mean(losses_iter_nn[k]), i) for k in range(len(losses_iter_nn))]
                [writer.add_scalar(f"losses_base/losses_iter_base{k}", np.mean(losses_iter_base[k]), i) for k in range(len(losses_iter_base))]


            losses = []
            losses_end = []
            time_diffs = []
            losses_iter = [[] for _ in range(args.deq_iter)]
            losses_iter_noQreg = [[] for _ in range(args.deq_iter)]
            losses_var = [[] for _ in range(args.deq_iter)]
            losses_iter_nocoeff = [[] for _ in range(args.deq_iter)]
            losses_proxy_iter_nocoeff = [[] for _ in range(args.deq_iter)]


            # print('nominal states: ', nominal_states)
            # print('nominal actions: ', nominal_actions)

    # torch.save(policy.state_dict(), "./model/bc_sac_pen")


if __name__ == "__main__":
    main()
