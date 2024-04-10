import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import qpth.qp_wrapper as ip_mpc
import qpth.AL_mpc as al_mpc
import qpth.al_utils as al_utils
import ipdb
# from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
# from policy_utils import SinusoidalPosEmb
import time

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

        self.fc_out = torch.nn.Linear(self.hdim, self.np * self.T)

        self.solver = self.anderson

    def forward(self, x):
        """
        compute the policy output for the given state x
        """
        xinp = self.fc_inp(x)
        xinp = self.ln_inp(xinp)
        z_shape = list(xinp.shape[:-1]) + [
            self.hdim,
        ]
        z = torch.zeros(z_shape).to(xinp)
        z_out = self.deq_fixed_point(xinp, z)
        x_ref = self.fc_out(z_out)
        x_ref = x_ref.view(-1, self.T, self.np)
        x_ref = x_ref + x[:, None, : self.np] * 10
        return x_ref

    def deq_fixed_point(self, x, z):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z, self.forward_res = self.solver(lambda z: self.f(z, x), z, **self.kwargs)
        z = self.f(z, x)

        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, x)

        def backward_hook(grad):
            g, self.backward_res = self.solver(
                lambda y: autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                grad,
                **self.kwargs
            )
            return g

        z.register_hook(backward_hook)
        return z

    def f(self, z, x):
        z = self.fcdeq1(z)
        z = self.reludeq1(z)
        z = self.lndeq1(z)
        out = self.lndeq3(self.reludeq2(z + self.lndeq2(x + self.fcdeq2(z))))
        return out

    def anderson(f, x0, m=5, lam=1e-4, max_iter=15, tol=1e-2, beta=1.0):
        """Anderson acceleration for fixed point iteration."""
        bsz, d, H, W = x0.shape
        X = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)
        F = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)
        X[:, 0], F[:, 0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
        X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].view_as(x0)).view(bsz, -1)

        H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
        H[:, 0, 1:] = H[:, 1:, 0] = 1
        y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
        y[:, 0] = 1

        res = []
        for k in range(2, max_iter):
            n = min(k, m)
            G = F[:, :n] - X[:, :n]
            H[:, 1 : n + 1, 1 : n + 1] = (
                torch.bmm(G, G.transpose(1, 2))
                + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[None]
            )
            alpha = torch.solve(y[:, : n + 1], H[:, : n + 1, : n + 1])[0][
                :, 1 : n + 1, 0
            ]  # (bsz x n)

            X[:, k % m] = (
                beta * (alpha[:, None] @ F[:, :n])[:, 0]
                + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
            )
            F[:, k % m] = f(X[:, k % m].view_as(x0)).view(bsz, -1)
            res.append(
                (F[:, k % m] - X[:, k % m]).norm().item()
                / (1e-5 + F[:, k % m].norm().item())
            )
            if res[-1] < tol:
                break
        return X[:, k % m].view_as(x0), res


# Questions:
# 1. What is a good input : As in a 'good' qualitative estimate of the error.
# - error between the trajectory spit out and the optimized trajectory
# - just x - x0
# - maybe also give velocities as input?
# - maybe also give the previous xref or (xref - x0) prediction as input? - especially important if we want to correct xref instead of just x (which is probably useful when there are other regularizers in the cost)
# - also should be compatible with recurrence on the latent z
# - ideally would like to merge it with the features or state (perhaps 3D) we extract from the image - to get a better sense of the error as input
# 2. What type of recurrence to add? beyond just the fixed point iterate.
# 3. How many QP solves to perform?
# 4. How to do the backward pass? - implicit differentiation of the fp network's fixed point? or just regular backprop? don't have any jacobians to compute fixed points, could compute the jacobians though
# 5. Unclear if deltas naively are the best output - regression is a hard problem especially at that precision.
# 6. Architecture - need to probably do some sort of GNN type thing. Using FFN to crunch entire trajectory seems suboptimal.
# 7. Also need to spit out weights corresponding to the steps the network is not confident about.
# 8. Diffusion gives a nice way to represent course signals when the network isn't confident about the output. To prevent overpenalizing the network for being uncertain.
#    Question is how to do that without explicitly introducing diffusion/stochasticity into the network.
#    (i) Perhaps just use the weights as a way to do that? This would imply we have to use the weights on the output loss as well.
#    (ii) This is however a broadly interesting question - If the weights or other mechanisms can be used to represent uncertainty well, then maybe diffusion is redundant?
#           - Question is what are those other mechanisms?
#           - CVAE type thing represents it in the latent space? (for the network to represent uncertainty aleotorically) => khai: CVAE handles epistemic uncertainty
#           - Dropout type thing represents it in the network parameters? (for the network to represent uncertainty epistemically) -- Similar to ensembles
#           - Predicting explicit weights represents it for the optimization problem? (for the optimization problem to represent uncertainty aleotorically)
#           - Using the weights in the loss as well - does something similar to CVAE? - but CVAE is probably more robust to overfitting and more principled
#           - Using the weights however is more interpretable and probably nicer optimization wise.
#           - Plus, the weights also aid the decoder training more explicitly - CVAE needs to be implicit thanks to the averaging over samples. - robustness v/s precision
#           - But come to think of it, CVAE and diffusion probably have quite a bit in common and there should be some obvious middle ground if we think harder.
#                 - In fact, feels like DEQ + CVAE is the natural way to go! - CVAE is a natural way to represent uncertainty in the latent space and DEQ is a natural way to
#    (iii) Diffusion v/s CVAE
#           - Diffusion is iterative and CVAE is not - makes diffusion representationally more powerful
#           - Diffusion reasons about uncertainy/noise more explicitly - CVAE is more implicit : Both have pros and cons
#                 - The explicitness makes each iteration more stable but also less powerful. Hence diffusion models are also huge in size and slow to train.
#           - The decoder objective in CVAE doesn't account for the uncertainty - diffusion does. This can sometimes lead to the CVAE decoder confidently spitting out blurry outputs.
#    (iv) Diffusion as a way to represent uncertainty in the optimization problem - can we use the normalized error in the estimates through the iterations as a way to represent uncertainty?
#    (v) In the diffusion case, should we just look at the overall setup as a way to add dynamics within diffusion models?
#           - or should we still look at it as an optimization problem with stochasticity added to aid with exploration.
#           - or instead the noise as just a means of incorporating uncertainty in the optimization steps at each iteration.
#           - The specific philosophy maybe doesn't matter too much - but it will probably guide the thinking. Some clarity on this would be nice!
# More important questions are probably still 1, 5, 6, 7
## TODOs:
# 1. Make input flexible - x, xref, xref - x0, xref - xref_prev, xref - xref_prev - x0, xref - xref_prev - x0 + v, etc.
# 2. Make outputs flexible - deltas, xref, xref - x0, etc. and also weights corresponding to the steps the network is not confident about.
# 3. Make architecture flexible - GNN or FFN or whatever
#       - Note : don't do parameter sharing between the nodes in gnn - the sequential order is important and maybe keeping the parameters somewhat separate is a good idea.
#       - but with limited data - parameter sharing might be a good idea - so maybe some sort of hybrid?
# 4. Make recurrence flexible - fixed point or whatever
# 5. Options for diffing through only fixed point, computing fixed point and other losses. 
#       - Only penalize the fixed point but train the network for the rest as well.
#       - Anderson/Broyden based fixed point solves
# 5. Figure out the collocation stuff 
#       - Is there a simpler/cleaner way to handle the initial deq iterations? - maybe we should try to satisfy the constraints exactly only if the network outputs are actually close enough to the constraint manifold
# 6. Complexity analysis
# 7. Write other solvers - Autmented lagrangian or ADMM or whatever
# 8. Confidences for each knot point - for the Q cost coefficient. - There should again probably be some TD component to the cost coefficient too. 
        

class DEQLayer(torch.nn.Module):
    def __init__(self, args, env):
        super().__init__()
        self.args = args
        self.nu = env.nu
        self.nx = env.nx
        self.np = args.np
        self.dt = env.dt
        self.T = args.T
        self.hdim = args.hdim
        self.layer_type = args.layer_type
        self.inp_type = ""  # args.inp_type
        self.out_type = ""  # args.out_type
        self.kernel_width = args.kernel_width
        self.pooling = args.pooling
        self.deq_expand = 4
        self.kernel_width_out = 1

        self.setup_input_layer()
        self.setup_deq_layer()
        self.setup_output_layer()

    def forward(self, x, z):
        """
        compute the policy output for the given state x
        """
        xinp = self.input_layer(x)
        z_out = self.deq_layer(xinp, z)
        dx_ref = self.output_layer(z_out)
        dx_ref = dx_ref.view(-1, self.T - 1, self.np)
        np = self.np//2
        vel_ref = dx_ref[..., np:]
        dx_ref = dx_ref[..., :np] * self.dt
        x_ref = torch.cat([dx_ref + x[:,None,:np], vel_ref], dim=-1)
        return x_ref, z_out

    def input_layer(self, x):
        if self.layer_type == "mlp":
            inp = self.inp_layer(x)
        elif self.layer_type == "gcn":
            t = self.time_emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
            x = x.reshape(-1, self.T, self.nx)
            x_emb = self.node_encoder(x)
            x0_emb = self.x0_encoder(x[:,0]).unsqueeze(1).repeat(1, self.T, 1)
            inp = torch.cat([x_emb, x0_emb, t], dim=-1)
            inp = self.input_encoder(inp)
        elif self.layer_type == "gat":
            NotImplementedError
        return inp
    def deq_layer(self, x, z):
        if self.layer_type == "mlp":
            z = self.fcdeq1(z)
            z = self.reludeq1(z)
            z = self.lndeq1(z)
            out = self.lndeq3(self.reludeq2(z + self.lndeq2(x + self.fcdeq2(z))))
        elif self.layer_type == "gcn":
            z = z.view(-1, self.T, self.hdim)
            z = self.convdeq1(z.permute(0, 2, 1))
            z = self.mishdeq1(z)
            z = self.gndeq1(z)
            out = self.gndeq3(self.mishdeq2(z + self.gndeq2(x + self.convdeq2(z))))
            out = out.permute(0, 2, 1).view(-1, self.hdim)
        elif self.layer_type == "gat":
            NotImplementedError
        return out

    def output_layer(self, z):
        if self.layer_type == "mlp":
            return self.out_layer(z)
        elif self.layer_type == "gcn":
            z = z.view(-1, self.T, self.hdim)
            z = self.convout(z.permute(0, 2, 1))
            z = self.mishout(z)
            z = self.gnout(z)
            return self.final_layer(z).permute(0, 2, 1)[:, 1:]
        elif self.layer_type == "gat":
            NotImplementedError

    def init_z(self, bsz):
        if self.layer_type == "mlp":
            return torch.zeros(bsz, self.hdim, dtype=torch.float32, device=self.args.device)
        elif self.layer_type == "gcn":
            return torch.zeros(bsz, self.T, self.hdim, dtype=torch.float32, device=self.args.device)
        elif self.layer_type == "gat":
            NotImplementedError

    def setup_input_layer(self):
        if self.layer_type == "mlp":
            # ipdb.set_trace()
            self.inp_layer = torch.nn.Sequential(
                torch.nn.Linear(self.nx + self.np * (self.T - 1), self.hdim),
                torch.nn.LayerNorm(self.hdim),
                # torch.nn.ReLU()
            )
            # self.fc_inp = torch.nn.Linear(self.nx + self.np*self.T, self.hdim)
            # self.ln_inp = torch.nn.LayerNorm(self.hdim)
        elif self.layer_type == "gcn":
            ## Get sinusoidal embeddings for the time steps
            # self.time_encoder = nn.Sequential(
            #     SinusoidalPosEmb(self.hdim),
            #     nn.Linear(self.hdim, self.hdim*4),
            #     nn.Mish(),
            #     nn.Linear(self.hdim*4, self.hdim),
            #     nn.LayerNorm(self.hdim)
            #     )
            self.time_emb = torch.nn.Parameter(torch.randn(self.T, self.hdim))
            ## Get the node embeddings
            self.node_encoder = nn.Sequential(
                nn.Linear(self.nx, self.hdim),
                nn.LayerNorm(self.hdim),
                nn.Mish()
                )
            
            self.x0_encoder = nn.Sequential(
                nn.Linear(self.nx, self.hdim),
                nn.LayerNorm(self.hdim),
                nn.Mish()
                )
            
            self.input_encoder = nn.Sequential(
                nn.Linear(self.hdim, self.hdim*4),
                nn.Mish(),
                nn.Linear(self.hdim*3, self.hdim),
                nn.LayerNorm(self.hdim),
                # nn.Mish()
                )
            
            self.global_pooling = {
                "max": torch.max,
                "mean": torch.mean,
                "sum": torch.sum
            }[self.pooling]
        elif self.layer_type == "gat":
            NotImplementedError

    def setup_deq_layer(
        self,
    ):
        if self.layer_type == "mlp":
            self.fcdeq1 = torch.nn.Linear(self.hdim, self.hdim)
            self.lndeq1 = torch.nn.LayerNorm(self.hdim)
            self.reludeq1 = torch.nn.ReLU()
            self.fcdeq2 = torch.nn.Linear(self.hdim, self.hdim)
            self.lndeq2 = torch.nn.LayerNorm(self.hdim)
            self.reludeq2 = torch.nn.ReLU()
            self.lndeq3 = torch.nn.LayerNorm(self.hdim)
        elif self.layer_type == "gcn":
            self.convdeq1 = torch.nn.Conv1d(self.hdim, self.hdim*self.deq_expand, self.kernel_width)
            self.convdeq2 = torch.nn.Conv1d(self.hdim*self.deq_expand, self.hdim, self.kernel_width)
            self.mishdeq1 = torch.nn.Mish()
            self.mishdeq2 = torch.nn.Mish()
            self.gndeq1 = torch.nn.GroupNorm(self.num_groups, self.hdim*self.deq_expand)
            self.gndeq2 = torch.nn.GroupNorm(self.num_groups, self.hdim)
            self.gndeq3 = torch.nn.GroupNorm(self.num_groups, self.hdim)
        elif self.layer_type == "gat":
            NotImplementedError

    def setup_output_layer(self):
        if self.layer_type == "mlp":
            self.out_layer = torch.nn.Sequential(
                torch.nn.Linear(self.hdim, self.np * (self.T - 1))
            )
        elif self.layer_type == "gcn":
            self.convout = torch.nn.Conv1d(self.hdim, self.hdim, self.kernel_width)
            self.gnout = torch.nn.GroupNorm(self.num_groups, self.hdim)
            self.mishout = torch.nn.Mish()
            self.final_layer = torch.nn.Conv1d(self.hdim, self.np, self.kernel_width_out)
        elif self.layer_type == "gat":
            NotImplementedError
        
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
        self.deq_iter = args.deq_iter
        self.model = DEQLayer(args, env)
        self.model.to(self.device)
        self.tracking_mpc = Tracking_MPC(args, env)
        self.mpc_time = []
        self.network_time = []

    def forward(self, x, x_gt, u_gt, mask, iter=0, qp_solve=True, lastqp_solve=False):
        """
        Run the DEQLayer and then run the MPC iteratively in a for loop until convergence
        Args:
            x (tensor 0 x bsz x nx): input state
        Returns:
            trajs (list of tuples): list of tuples of nominal states and actions for each deq iteration
            trajs[k][0] (tensor T x bsz x nx): nominal states for the kth deq iteration
            trajs[k][1] (tensor T x bsz x nu): nominal actions for the kth deq iteration
        """
        # initialize trajectory with current state
        x_ref = torch.cat([x]*self.T, dim=-1).detach().clone()
        # x_ref = x_gt.view(x_ref.shape)
        z = self.model.init_z(x.shape[0]).to(self.device)

        # initialize trajs list
        trajs = []
        bsz = x.shape[0]
        if self.args.solver_type == "al":
            self.tracking_mpc.reinitialize(x, mask[:, :, None])
        

        # run the DEQ layer for deq_iter iterations
        for i in range(self.deq_iter):
            # torch.cuda.synchronize()
            # start = time.time()
            # x_ref = (x_ref.view(bsz, self.T, -1)*mask[:, :, None]).view(bsz, -1)
            x_ref, z = self.model(x_ref, z)
            x_ref = x_ref.view(-1, self.T-1, self.np)
            x_ref = torch.cat([x[:, None, :], x_ref], dim=1)
            # nominal_states = x_ref.transpose(0, 1)
            # nominal_actions = torch.zeros_like(nominal_states[..., :self.nu])
            # trajs.append((nominal_states, nominal_actions))
            # x_ref = x_ref.reshape(bsz, -1)
            
            # ipdb.set_trace()
            # x_ref = x_gt + x_ref - x_ref.detach().clone()
            xu_ref = torch.cat(
                [x_ref, torch.zeros_like(x_ref[..., :self.nu])], dim=-1
            )
            x_ref_tr = x_ref
            u_ref_tr = torch.zeros_like(x_ref_tr[..., :self.nu])#u_gt.transpose(0, 1)
            nominal_states = x_ref
            nominal_actions = torch.zeros_like(nominal_states[..., :self.nu])
            # torch.cuda.synchronize()
            # end = time.time()
            # self.network_time.append(end-start)
            if qp_solve:
                # torch.cuda.synchronize()
                # start = time.time()
                # ipdb.set_trace()
                nominal_states, nominal_actions = self.tracking_mpc(x, xu_ref, x_ref_tr, u_ref_tr)
                
                # torch.cuda.synchronize()
                # end = time.time()
                # self.mpc_time.append(end-start)
            nominal_states_net = x_ref#.transpose(0, 1)
            trajs.append((nominal_states_net, nominal_states, nominal_actions))
            # x_ref = nominal_states_net.transpose(0, 1).reshape(bsz, -1)#.detach().clone().reshape(bsz, -1)
            x_ref = nominal_states.reshape(bsz, -1).detach().clone()
        # print(f"Network time: {np.mean(self.network_time)} MPC time: {np.mean(self.mpc_time)}")
        self.network_time = []
        self.mpc_time = []
        if lastqp_solve:
            nominal_states, nominal_actions = self.tracking_mpc(x, xu_ref, x_ref_tr, u_ref_tr)
            trajs[-1] = (nominal_states_net, nominal_states, nominal_actions)
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
        self.dyn = env.dynamics
        self.dyn_jac = env.dynamics_derivatives

        # May comment out input constraints for now
        self.device = args.device
        self.u_upper = torch.tensor(env.action_space.high).to(self.device)
        self.u_lower = torch.tensor(env.action_space.low).to(self.device)
        self.qp_iter = args.qp_iter
        self.eps = args.eps
        self.warm_start = args.warm_start
        self.bsz = args.bsz

        self.Q = args.Q.to(self.device)
        self.R = args.R.to(self.device)
        self.dtype = torch.float64 if args.dtype=="double" else torch.float32
        # self.Qf = args.Qf
        if args.Q is None:
            self.Q = torch.ones(self.nx, dtype=self.dtype, device=self.device)
            # self.Qf = torch.ones(self.nx, dtype=torch.float32, device=self.device)
            self.R = torch.ones(self.nu, dtype=self.dtype, device=self.device)
        self.Q = torch.cat([self.Q, self.R], dim=0).to(self.dtype)
        self.Q = torch.diag(self.Q).repeat(self.bsz, self.T, 1, 1)

        self.u_init = torch.randn(
            self.bsz, self.T, self.nu, dtype=self.dtype, device=self.device
        )
        
        self.single_qp_solve = True if self.qp_iter == 1 else False

        if args.solver_type == "al":
            self.ctrl = al_mpc.MPC(
                self.nx,
                self.nu,
                self.T,
                u_lower=self.u_lower,
                u_upper=self.u_upper,
                # al_iter=self.qp_iter,
                exit_unconverged=False,
                eps=1e-5,
                n_batch=self.bsz,
                backprop=False,
                verbose=0,
                u_init=self.u_init,
                solver_type="dense",
                dtype=self.dtype,
            )
        else:
            self.ctrl = ip_mpc.MPC(
                self.nx,
                self.nu,
                self.T,
                u_lower=self.u_lower,
                u_upper=self.u_upper,
                qp_iter=self.qp_iter,
                exit_unconverged=False,
                eps=1e-5,
                n_batch=self.bsz,
                backprop=False,
                verbose=0,
                u_init=self.u_init.transpose(0,1),
                grad_method=ip_mpc.GradMethods.ANALYTIC,
                solver_type="dense",
                single_qp_solve=self.single_qp_solve,
            )

    def forward(self, x0, xu_ref, x_ref, u_ref):
        """
        compute the mpc output for the given state x and reference x_ref
        """
        if self.args.solver_type == "al":
            xu_ref = torch.cat([x_ref, u_ref], dim=-1)
            if self.x_init is None:
                self.x_init = self.ctrl.x_init = x_ref
                self.u_init = self.ctrl.u_init = u_ref

        self.compute_p(xu_ref)
        if self.args.solver_type == "al":
            cost = al_utils.QuadCost(self.Q, self.p)
        else:
            cost = ip_mpc.QuadCost(self.Q.transpose(0,1), self.p.transpose(0,1))
            self.ctrl.u_init = self.u_init.transpose(0,1)

        state = x0  # .unsqueeze(0).repeat(self.bsz, 1)
        nominal_states, nominal_actions = self.ctrl(state, cost, self.dyn, self.dyn_jac)
        if self.args.solver_type == "ip":
            nominal_states = nominal_states.transpose(0, 1)
            nominal_actions = nominal_actions.transpose(0, 1)
        # ipdb.set_trace()
        self.u_init = nominal_actions.clone().detach()
        return nominal_states, nominal_actions

    def compute_p(self, x_ref):
        """
        compute the p for the quadratic objective using self.Q as the diagonal matrix and the reference x_ref at each time without a for loop
        """
        # self.p = torch.zeros(
        #     self.T, self.bsz, self.nx + self.nu, dtype=torch.float32, device=self.device
        # )
        # self.p[:, :, : self.nx] = -(
        #     self.Q[:, :, : self.nx, : self.nx] * x_ref.unsqueeze(-2)
        # ).sum(dim=-1)
        self.p = -(self.Q * x_ref.unsqueeze(-2)).sum(dim=-1)
        return self.p
    
    def reinitialize(self, x, mask):
        self.u_init = torch.randn(
            self.bsz, self.T, self.nu, dtype=x.dtype, device=x.device
        )
        self.x_init = None
        self.ctrl.reinitialize(x, mask)

class NNMPCPolicy(torch.nn.Module):
    """
    Feedforward Neural network based MPC policy
    """

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


class NNPolicy(torch.nn.Module):
    """
    Some NN-based policy trained with behavioral cloning, outputting a trajectory (state or input) of horizon T
    """

    def __init__(self, args, env):
        super().__init__()
        self.args = args
        self.nu = env.nu
        self.nx = env.nx
        self.np = args.np
        self.T = args.T
        self.dt = env.dt
        self.device = args.device
        self.hdim = args.hdim
        self.output_type = 0
        # output_type = 0 : output only actions
        # output_type = 1 : output only states
        # output_type = 2 : output states and actions
        # output_type = 3 : output only positions

        # define the network layers :
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.nx, self.hdim),
            torch.nn.LayerNorm(self.hdim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hdim, self.hdim),
            torch.nn.LayerNorm(self.hdim),
            torch.nn.ReLU(),
        )
        if self.output_type == 0:
            self.out_dim = self.nu * self.T
        elif self.output_type == 1:
            self.out_dim = self.nx * self.T
        elif self.output_type == 2:
            self.out_dim = (self.nx + self.nu) * self.T
        elif self.output_type == 3:
            self.out_dim = (self.np) * self.T

        self.model.add_module("out", torch.nn.Linear(self.hdim, self.out_dim))

    def forward(self, x):
        """
        compute the trajectory given state x
        Args:
            x (tensor bsz x nx): input state
        Returns:
            states (tensor bsz x T x nx): nominal states or None
            actions (tensor bsz x T x nu): nominal actions or None
        """
        if self.output_type == 0:
            actions = self.model(x)
            actions = actions.view(-1, self.T, self.nu)
            states = None
        elif self.output_type == 1:
            states = self.model(x)
            states = states.view(-1, self.T, self.nx)
            actions = None
        elif self.output_type == 2:
            states = self.model(x)[:, : self.nx * self.T]
            states = states.view(-1, self.T, self.nx)
            actions = self.model(x)[:, self.nx * self.T :]
            actions = actions.view(-1, self.T, self.nu)
        elif self.output_type == 3:
            pos = self.model(x)
            vel = (pos[:, 1:] - pos[:, :-1]) / self.dt
            vel = torch.cat([vel, vel[:, -1:]], dim=1)
            states = torch.cat([pos, vel], dim=-1).view(-1, self.T, self.nx)
            actions = None
        return states, actions
