
import numpy as np
import torch
import torch.autograd as autograd
import qpth.qp_wrapper as mpc
import ipdb
from envs import PendulumEnv, PendulumDynamics

class DEQPolicy(torch.nn.Module):
    def __init__(self, args, env):
        super().__init__()
        self.args = args
        self.nu = env.nu
        self.nx = env.nx
        self.dt = env.dt
        self.T = args.T
        self.hdim = args.hdim

        self.fc_inp = torch.nn.Linear(self.nx, self.hdim)
        self.ln_inp = torch.nn.LayerNorm(self.hdim)

        self.fcdeq1 = torch.nn.Linear(self.hdim, self.hdim)
        self.lndeq1 = torch.nn.LayerNorm(self.hdim)
        self.reludeq1 = torch.nn.ReLU()
        self.fcdeq2 = torch.nn.Linear(self.hdim, self.hdim)
        self.lndeq2 = torch.nn.LayerNorm(self.hdim)
        self.reludeq2 = torch.nn.ReLU()
        self.lndeq3 = torch.nn.LayerNorm(self.hdim)

        self.fc_out = torch.nn.Linear(self.hdim, self.np*self.T)

        self.solver = self.anderson
    
    def forward(self, x):
        """
        compute the policy output for the given state x
        """
        xinp = self.fc_inp(x)
        xinp = self.ln_inp(xinp)
        z_shape = list(xinp.shape[:-1]) + [self.hdim,]
        z = torch.zeros(z_shape).to(xinp)
        z_out = self.deq_fixed_point(xinp, z)
        x_ref = self.fc_out(z_out)
        x_ref = x_ref.view(-1, self.T, self.np)
        x_ref = x_ref + x[:,None,:self.np]*10
        return x_ref
    
    def deq_fixed_point(self, x, z):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z, self.forward_res = self.solver(lambda z : self.f(z, x), z, **self.kwargs)
        z = self.f(z,x)
        
        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0,x)
        def backward_hook(grad):
            g, self.backward_res = self.solver(lambda y : autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                               grad, **self.kwargs)
            return g

        z.register_hook(backward_hook)
        return z
    
    def f(self, z, x):
        z = self.fcdeq1(z)
        z = self.reludeq1(z)
        z = self.lndeq1(z)
        out = self.lndeq3(self.reludeq2(z + self.lndeq2(x + self.fcdeq2(z))))
        return out

    def anderson(f, x0, m=5, lam=1e-4, max_iter=15, tol=1e-2, beta = 1.0):
        """ Anderson acceleration for fixed point iteration. """
        bsz, d, H, W = x0.shape
        X = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
        F = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
        X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
        X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)
        
        H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
        H[:,0,1:] = H[:,1:,0] = 1
        y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
        y[:,0] = 1
        
        res = []
        for k in range(2, max_iter):
            n = min(k, m)
            G = F[:,:n]-X[:,:n]
            H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
            alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]   # (bsz x n)
            
            X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
            F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
            res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
            if (res[-1] < tol):
                break
        return X[:,k%m].view_as(x0), res

# Questions:
    # 1. What is a good input : As in a 'good' qualitative estimate of the error. 
        # - error between the trajectory spit out and the optimized trajectory
        # - just xref - x0 - maybe also give velocities as input?
        # - also should be compatible with recurrence on the latent z
    # 2. What type of recurrence to add? beyond just the fixed point iterate. 
    # 3. How many QP solves to perform? 
    # 4. How to do the backward pass? - implicit differentiation of the fp network's fixed point? or just regular backprop? don't have any jacobians to compute fixed points, could compute the jacobians though
    # 5. Unclear if deltas naively are the best output - regression is a hard problem especially at that precision. 
    # 6. Architecture - need to probably do some sort of GNN type thing. Using FFN to crunch entire trajectory seems suboptimal.
    # 7. Also need to spit out weights corresponding to the steps the network is not confident about. 
# More important questions are probably still 1, 5, 6
class DEQLayer(torch.nn.Module):
    def __init__(self, args, env):
        super().__init__()
        self.args = args
        self.nu = env.nu
        self.nx = env.nx
        self.np = args.np
        self.T = args.T
        self.hdim = args.hdim

        self.fc_inp = torch.nn.Linear(self.nx + self.np*self.T, self.hdim)
        self.ln_inp = torch.nn.LayerNorm(self.hdim)

        self.fcdeq1 = torch.nn.Linear(self.hdim, self.hdim)
        self.lndeq1 = torch.nn.LayerNorm(self.hdim)
        self.reludeq1 = torch.nn.ReLU()
        self.fcdeq2 = torch.nn.Linear(self.hdim, self.hdim)
        self.lndeq2 = torch.nn.LayerNorm(self.hdim)
        self.reludeq2 = torch.nn.ReLU()
        self.lndeq3 = torch.nn.LayerNorm(self.hdim)

        self.fc_out = torch.nn.Linear(self.hdim, self.np*self.T)

    def forward(self, x, z):
        """
        compute the policy output for the given state x
        """
        xinp = self.fc_inp(x)
        xinp = self.ln_inp(xinp)
        # z_shape = list(xinp.shape[:-1]) + [self.hdim,]
        # z = torch.zeros(z_shape).to(xinp)
        z_out = self.deq_layer(xinp, z)
        dx_ref = self.fc_out(z_out)
        dx_ref = dx_ref.view(-1, self.T, self.np)
        # x_ref = dx_ref + x[:,None,:self.np]*10
        return dx_ref, z_out

    def deq_layer(self, x, z):
        z = self.fcdeq1(z)
        z = self.reludeq1(z)
        z = self.lndeq1(z)
        out = self.lndeq3(self.reludeq2(z + self.lndeq2(x + self.fcdeq2(z))))
        return out

class DEQMPCPolicy(torch.nn.Module):
    def __init__(self, args, env):
        super().__init__()
        self.args = args
        self.nu = env.nu
        self.nx = env.nx
        self.np = args.np
        self.T = args.T
        self.dt = env.dt
        self.device = args.device
        self.model_layer = DEQLayer(args, env)
        self.model_layer.to(self.device)
        self.tracking_mpc = Tracking_MPC(args, env)
    
    def forward(self, x):
        """
        Run the DEQLayer and then run the MPC iteratively in a for loop until convergence
        """
        # initialize trajectory with zeros
        dx_ref = torch.zeros(x.shape[0], self.T, self.np).to(self.device)

        # initialize trajs list
        trajs = []

        # run the DEQ layer for deq_iter iterations
        for i in range(self.deq_iter):
            x_ref = torch.cat([x[:,None,:self.np], x[:,None,:self.np] + dx_ref], dim=1)
            dx_ref = self.model(x_ref)
            dx_ref = dx_ref.view(-1, self.T, self.np)

            dx_ref = torch.cat(
                [
                    dx_ref,
                    torch.zeros(
                        list(dx_ref.shape[:-1])
                        + [
                            self.np,
                        ]
                    ).to(self.args.device),
                ],
                dim=-1,
            ).transpose(0, 1)
            nominal_states, nominal_actions = self.tracking_mpc(x, dx_ref)
            dx_ref = nominal_states.transpose(0, 1) - x[:,None,:self.np]
            trajs.append((nominal_states, nominal_actions))

        return trajs

    
class FFDNetwork(torch.nn.Module):
    """
    Feedforward network to generate reference trajectories of horizon T
    """

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
        self.fc3 = torch.nn.Linear(256, self.np * self.T)
        self.net = torch.nn.Sequential(
            self.fc1, self.ln1, self.relu1, self.fc2, self.ln2, self.relu2, self.fc3
        )

    def forward(self, x):
        """
        compute the policy output for the given state x
        """
        dx_ref = self.net(x)
        dx_ref = dx_ref.view(-1, self.T, self.np)
        x_ref = dx_ref + x[:, None, : self.np]
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
        self.u_upper = None  # torch.tensor(env.action_space.high).to(args.device)
        self.u_lower = None  # torch.tensor(env.action_space.low).to(args.device)
        self.qp_iter = args.qp_iter
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

        self.u_init = torch.zeros(
            self.T, self.bsz, self.nu, dtype=torch.float32, device=self.device
        )

        self.ctrl = mpc.MPC(
            self.nx,
            self.nu,
            self.T,
            u_lower=self.u_lower,
            u_upper=self.u_upper,
            qp_iter=self.qp_iter,
            exit_unconverged=False,
            eps=1e-2,
            n_batch=self.bsz,
            backprop=False,
            verbose=0,
            u_init=self.u_init,
            grad_method=mpc.GradMethods.AUTO_DIFF,
            solver_type="dense",
        )

    def forward(self, x_init, x_ref):
        """
        compute the mpc output for the given state x and reference x_ref
        """

        self.compute_p(x_ref)
        cost = mpc.QuadCost(self.Q, self.p)
        self.ctrl.u_init = self.u_init
        state = x_init  # .unsqueeze(0).repeat(self.bsz, 1)
        nominal_states, nominal_actions = self.ctrl(state, cost, PendulumDynamics())
        return nominal_states, nominal_actions

    def compute_p(self, x_ref):
        """
        compute the p for the quadratic objective using self.Q as the diagonal matrix and the reference x_ref at each time without a for loop
        """
        self.p = torch.zeros(
            self.T, self.bsz, self.nx + self.nu, dtype=torch.float32, device=self.device
        )
        self.p[:, :, : self.nx] = -(
            self.Q[:, :, : self.nx, : self.nx] * x_ref.unsqueeze(-2)
        ).sum(dim=-1)
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
        x_ref = torch.cat(
            [
                x_ref,
                torch.zeros(
                    list(x_ref.shape[:-1])
                    + [
                        self.np,
                    ]
                ).to(self.args.device),
            ],
            dim=-1,
        ).transpose(0, 1)
        nominal_states, nominal_actions = self.tracking_mpc(x, x_ref)
        return nominal_states, nominal_actions


# class DEQPolicy:

# class DEQMPCPolicy:

# class NNPolicy:

# class NNMPCPolicy:

