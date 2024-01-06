import math
import time

import numpy as np
import torch
import torch.autograd as autograd
import qpth.qp_wrapper as mpc
import ipdb
from envs import PendulumEnv, PendulumDynamics
from datagen import get_gt_data, merge_gt_data, sample_trajectory
## example task : hard pendulum with weird coordinates to make sure direct target tracking is difficult

class FFDNetwork(torch.nn.Module):
    def __init__(self, args, env):
        super().__init__()
        self.args = args
        self.nu = env.nu
        self.nx = env.nx
        self.np = args.np
        self.T = args.T


        ## define the network layers : 
        self.fc1 = torch.nn.Linear(self.nx, 256)
        self.ln1 = torch.nn.LayerNorm(256)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(256, 256)
        self.ln2 = torch.nn.LayerNorm(256)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(256, self.np*self.T)
        self.net = torch.nn.Sequential(self.fc1, self.ln1, self.relu1, self.fc2, self.ln2, self.relu2, self.fc3)

    def forward(self, x):
        """
        compute the policy output for the given state x
        """
        x_ref = self.net(x)
        x_ref = x_ref.view(-1, self.T, self.np)
        x_ref = x_ref + x[:,None,:self.np]*10
        return x_ref


class Tracking_MPC(torch.nn.Module):
    def __init__(self, args, env):
        super().__init__()
        self.args = args
        self.nu = env.nu
        self.nx = env.nx
        self.dt = env.dt
        self.T = args.T

        # May comment out input constraints for now
        self.u_upper = torch.tensor(env.action_space.high).to(args.device)  
        self.u_lower = torch.tensor(env.action_space.low).to(args.device)
        self.max_iter = args.max_iter
        self.eps = args.eps
        self.warm_start = args.warm_start
        self.bsz = args.bsz
        self.device = args.device

        self.Q = args.Q
        self.R = args.R
        # self.Qf = args.Qf
        if args.Q is None:
            self.Q = torch.ones(self.nx, dtype=torch.float32, device=args.device)
            # self.Qf = torch.ones(self.nx, dtype=torch.float32, device=args.device)
            self.R = torch.ones(self.nu, dtype=torch.float32, device=args.device)
        self.Q = torch.cat([self.Q, self.R], dim=0)
        self.Q = torch.diag(self.Q).repeat(self.T, self.bsz, 1, 1)

        self.u_init = torch.zeros(self.T, self.bsz, self.nu, dtype=torch.float32, device=self.device)

        self.ctrl = mpc.MPC(self.nx, self.nu, self.T, u_lower=self.u_lower, u_upper=self.u_upper, qp_iter=self.max_iter,
                        exit_unconverged=False, eps=1e-2,
                        n_batch=self.bsz, backprop=False, verbose=0, u_init=self.u_init,
                        grad_method=mpc.GradMethods.AUTO_DIFF, solver_type='dense')

    def forward(self, x_init, x_ref):
        """
        compute the mpc output for the given state x and reference x_ref
        """

        self.compute_p(x_ref)
        cost = mpc.QuadCost(self.Q, self.p)
        self.ctrl.u_init = self.u_init
        state = x_init#.unsqueeze(0).repeat(self.bsz, 1)
        nominal_states, nominal_actions = self.ctrl(state, cost, PendulumDynamics())
        return nominal_states, nominal_actions

    def compute_p(self, x_ref):
        """
        compute the p for the quadratic objective using self.Q as the diagonal matrix and the reference x_ref at each time without a for loop
        """
        self.p = torch.zeros(self.T, self.bsz, self.nx+self.nu, dtype=torch.float32, device=self.device)
        self.p[:, :, :self.nx] = -(self.Q[:, :, :self.nx, :self.nx] * x_ref.unsqueeze(-2)).sum(dim=-1)
        return self.p



class NNMPCPolicy(torch.nn.Module):
    def __init__(self, args, env):
        super().__init__()
        self.args = args
        self.nu = env.nu
        self.nx = env.nx
        self.np = args.np
        self.T = args.T
        self.dt = env.dt
        self.device = args.device
        self.model = FFDNetwork(args, env)
        self.model.to(self.device)
        self.tracking_mpc = Tracking_MPC(args, env)
    
    def forward(self, x):
        """
        compute the policy output for the given state x
        """
        x_ref = self.model(x)
        # x_ref = x_ref.view(-1, self.np)
        x_ref = torch.cat([x_ref, torch.zeros(list(x_ref.shape[:-1])+[self.np,]).to(self.args.device)], dim=-1).transpose(0,1)
        nominal_states, nominal_actions = self.tracking_mpc(x, x_ref)
        return nominal_states, nominal_actions

# class DEQPolicy:

# class DEQMPCPolicy:

# class NNPolicy:

# class NNMPCPolicy:
    
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--np', type=int, default=1)
    parser.add_argument('--T', type=int, default=5)
    # parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument('--eps', type=float, default=1e-2)
    parser.add_argument('--warm_start', type=bool, default=True)
    parser.add_argument('--bsz', type=int, default=80)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    env = PendulumEnv(stabilization=True)
    gt_trajs = get_gt_data(args, env)

    gt_trajs = merge_gt_data(gt_trajs)
    
    args.Q = torch.Tensor([10., 0.001]).to(args.device)
    args.R = torch.Tensor([0.001]).to(args.device)
    policy = NNMPCPolicy(args, env).to(args.device)
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=1e-4)
    

    # run imitation learning using gt_trajs
    for i in range(1000):
        # sample bsz random trajectories from gt_trajs and a random time step for each
        traj_sample = sample_trajectory(gt_trajs, args.bsz, args.T)
        traj_sample = {k: v.to(args.device) for k, v in traj_sample.items()}
        # ipdb.set_trace()
        traj_sample["state"] = torch.cat([traj_sample["state"], PendulumDynamics()(traj_sample["state"][:,-1], traj_sample["action"][:,-1])[:,None]], dim=1)
        # idxs = np.random.randint(0, len(gt_trajs), args.bsz)
        # t_idxs = np.random.randint(0, len(gt_trajs[0]), args.bsz)
        # x_init = gt_trajs[idxs, t_idxs]
        # x_gt = gt_trajs[idxs, t_idxs:t_idxs+args.T]

        # ipdb.set_trace()
        nominal_states, nominal_actions = policy(traj_sample["state"][:,0])
        loss = torch.abs((nominal_states.transpose(0,1) - traj_sample["state"][:, 1:])*traj_sample["mask"][:,:,None]).sum(dim=-1).mean() #+ torch.norm(nominal_actions)
        # ipdb.set_trace()
        # loss += (nominal_actions[0] - traj_sample["action"][:, 0]).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        # gradient clipping
        # torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 4)
        optimizer.step()
        if i % 100 == 0:
            print('iter: ', i)
            print("grad norm: ", torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1000))
            print('loss: ', loss)
            # print('nominal states: ', nominal_states)
            # print('nominal actions: ', nominal_actions)





if __name__ == '__main__':
    main()