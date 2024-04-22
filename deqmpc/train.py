import utils
from torch.utils.tensorboard import SummaryWriter
from policies import NNMPCPolicy, DEQPolicy, DEQMPCPolicy, NNPolicy
from datagen import get_gt_data, merge_gt_data, sample_trajectory
from rex_quadrotor import RexQuadrotor
from my_envs.cartpole import CartpoleEnv
from envs import PendulumEnv, PendulumDynamics, IntegratorEnv, IntegratorDynamics
from envs_v1 import CartpoleV0Env

import ipdb
import qpth.qp_wrapper as mpc
import math
import time

import numpy as np
import torch
import torch.autograd as autograd
import sys
import os
import time
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, project_dir)

# import tensorboard from pytorch

# example task : hard pendulum with weird coordinates to make sure direct target tracking is difficult


def seeding(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="pendulum")
    parser.add_argument("--np", type=int, default=1)
    parser.add_argument("--T", type=int, default=5)
    # parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument("--qp_iter", type=int, default=1)
    parser.add_argument("--eps", type=float, default=1e-2)
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
    parser.add_argument("--data_noise_type", type=int, default=0)  # check noise_utils.py for noise_type
    parser.add_argument("--data_noise_std", type=float, default=0.01)
    parser.add_argument("--data_noise_mean", type=float, default=0.0)

    args = parser.parse_args()
    seeding(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device if args.device is None else args.device
    if args.save:
        if not os.path.exists("./logs/" + args.name):
            os.makedirs("./logs/" + args.name)
        args.name = args.name + \
            f"_T{args.T}_bsz{args.bsz}_deq_iter{args.deq_iter}_np{args.np}"
        writer = SummaryWriter("./logs/" + args.name)

    kwargs = {"dtype": torch.float64,
              "device": args.device, "requires_grad": False}
    if args.env == "pendulum":
        env = PendulumEnv(stabilization=False)
        gt_trajs = get_gt_data(args, env, "sac")
    elif args.env == "integrator":
        env = IntegratorEnv()
        gt_trajs = get_gt_data(args, env, "sac")
    elif args.env == "rexquadrotor":
        env = RexQuadrotor(bsz=args.bsz, device=args.device)
        gt_trajs = get_gt_data(args, env, "cgac")
    elif args.env == "pendulum_stabilize":
        env = PendulumEnv(stabilization=True)
        gt_trajs = get_gt_data(args, env, "sac")
    elif args.env == "cartpole-v0":
        env = CartpoleV0Env()
        gt_trajs = get_gt_data(args, env, "cgac")
    elif args.env == "cartpole1link":
        env = CartpoleEnv(nx=4, dt=0.05, stabilization=False, kwargs=kwargs)
        gt_trajs = get_gt_data(args, env, "cgac")
    elif args.env == "cartpole2link":
        env = CartpoleEnv(nx=6, dt=0.03, stabilization=False, kwargs=kwargs)
        gt_trajs = get_gt_data(args, env, "cgac")

    # gt_trajs = get_gt_data(args, env, "mpc")
    # gt_trajs = get_gt_data(args, env, "cgac")
    gt_trajs = merge_gt_data(gt_trajs)
    args.Q = env.Qlqr.to(args.device)
    args.R = env.Rlqr.to(args.device)
    if args.deq:
        policy = DEQMPCPolicy(args, env).to(args.device)
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

    # run imitation learning using gt_trajs
    for i in range(20000):
        # sample bsz random trajectories from gt_trajs and a random time step for each
        traj_sample = sample_trajectory(gt_trajs, args.bsz, args.T)
        traj_sample = {k: v.to(args.device) for k, v in traj_sample.items()}

        if args.env == "pendulum":
            traj_sample["state"] = utils.unnormalize_states_pendulum(
                traj_sample["state"])
        elif args.env == "cartpole1link" or args.env == "cartpole2link":
            traj_sample["state"] = utils.unnormalize_states_cartpole_nlink(
                traj_sample["state"])
        iter_qp_solve = False if (i < 5000 and args.pretrain) else True
        # warm start only after 1000 iterations
        qp_solve = iter_qp_solve and args.qp_solve
        lastqp_solve = args.lastqp_solve and iter_qp_solve
        if args.deq:
            loss = 0.0
            start = time.time()
            gt_states = utils.corrupt_observation(
                traj_sample["state"], args.data_noise_type, args.data_noise_std, args.data_noise_mean)
            gt_state0 = gt_states[:, 0]
            traj_mask = traj_sample["mask"]
            trajs, dyn_res = policy(gt_state0, gt_states, traj_sample["action"],
                                    traj_mask, iter=i, qp_solve=qp_solve, lastqp_solve=lastqp_solve)
            end = time.time()
            dyn_resids.append(dyn_res)
            for j, (nominal_states_net, nominal_states, nominal_actions) in enumerate(trajs):
                loss_j = (torch.abs((nominal_states - gt_states) *
                          traj_mask[:, :, None]).sum(dim=-1).mean())
                loss += loss_j
            loss_end = (torch.abs((nominal_states - gt_states) *
                        traj_mask[:, :, None]).sum(dim=-1).mean())
        else:
            loss = 0.0
            nominal_states, nominal_actions = policy(
                traj_sample["state"][:, 0])
            if policy.output_type == 0 or policy.output_type == 2:
                loss += (
                    torch.abs((nominal_actions - traj_sample["action"]) * traj_sample["mask"][:, :, None]).sum(dim=-1).mean())
            if policy.output_type == 1 or policy.output_type == 2 or policy.output_type == 3:
                loss += (torch.abs(
                    # [:, :, : policy.np]
                        (nominal_states - traj_sample["state"]) * traj_sample["mask"][:, :, None]).sum(dim=-1).mean())
            loss_end = torch.Tensor([0.0])
        time_diffs.append(end-start)
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        losses_end.append(loss_end.item())
        # gradient clipping
        # torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 4)
        optimizer.step()
        if i % 100 == 0:
            print("iter: ", i)
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

            losses = []
            losses_end = []
            time_diffs = []
            # print('nominal states: ', nominal_states)
            # print('nominal actions: ', nominal_actions)

    # torch.save(policy.state_dict(), "./model/bc_sac_pen")

if __name__ == "__main__":
    main()