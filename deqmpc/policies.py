import numpy as nq
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
from deq_layer import *


# POSSIBLE OUTPUT TYPES OF THE POLICY
# 0: horizon action
# 1: horizon state
# 2: horizon state + action
# 3: horizon config (state no vel)



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
# TODOs:
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

class DEQMPCPolicy(torch.nn.Module):
    def __init__(self, args, env):
        super().__init__()
        self.args = args
        self.nu = env.nu
        self.nx = env.nx
        self.nq = args.nq
        self.T = args.T
        self.dt = env.dt
        self.bsz = args.bsz
        self.deq_reg = args.deq_reg
        self.device = args.device
        self.deq_iter = args.deq_iter
        self.model = DEQLayer(args, env)  #TODO different types
        self.model.to(self.device)
        self.out_type = args.policy_out_type  # output type of policy
        self.loss_type = args.loss_type  # loss type for policy
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
        x_ref = x_ref.view(-1, self.T, self.nx)
        nominal_actions = torch.zeros((x.shape[0], self.T, self.nu), device=self.device)
        # x_ref = x_gt.view(x_ref.shape)
        z = self.model.init_z(self.bsz).to(self.device)
        # ipdb.set_trace()
        out_aux_dict = {"z": z, "x": x_ref, 'u': nominal_actions}

        if self.args.solver_type == "al":
            self.tracking_mpc.reinitialize(x, mask[:, :, None])
    
        # run the DEQ layer for deq_iter iterations
        trajs, dyn_res = self.deqmpc_iter(x, out_aux_dict, x_gt, u_gt, mask, qp_solve, lastqp_solve)        
        return trajs, dyn_res

    def deqmpc_iter(self, obs, out_aux_dict, x_gt, u_gt, mask, qp_solve=False, lastqp_solve=False): 
        if (qp_solve):
            deq_iter = self.deq_iter * 2
        else:
            deq_iter = self.deq_iter   

        trajs = []
        for i in range(deq_iter):
            in_obs_dict = {"o": obs}
            # ipdb.set_trace()
            out_mpc_dict, out_aux_dict = self.model(in_obs_dict, out_aux_dict)
            x_t, x_ref, u_ref = out_mpc_dict["x_t"], out_mpc_dict["x_ref"], out_mpc_dict["u_ref"]
            xu_ref = torch.cat([x_ref, u_ref], dim=-1)
            nominal_states_net = x_ref
            nominal_states = nominal_states_net
            nominal_actions = u_ref
            # ipdb.set_trace()
            if qp_solve and i > self.deq_iter:
                # ipdb.set_trace()
                nominal_states, nominal_actions = self.tracking_mpc(x_t, xu_ref, x_ref, u_ref)
                out_aux_dict["x"] = nominal_states.detach().clone()
                out_aux_dict["u"] = nominal_actions.detach().clone()
            if not lastqp_solve:
                out_aux_dict["x"] = out_aux_dict["x"].detach().clone()
            if not lastqp_solve:
                trajs.append((nominal_states_net, nominal_states, nominal_actions))
            else:
                trajs.append((nominal_states_net.detach().clone(), nominal_states.detach().clone(), nominal_actions.detach().clone()))

        dyn_res = (self.tracking_mpc.dyn(x_gt[:, :-1].reshape(-1, self.nx).double(
        ), u_gt[:, :-1].reshape(-1, self.nu).double()) - x_gt[:,1:].reshape(-1, self.nx)).reshape(self.bsz, -1).norm(dim=1).mean().item()
        self.network_time = []
        self.mpc_time = []

        if lastqp_solve:
            nominal_states, nominal_actions = self.tracking_mpc(
                x_t, xu_ref, x_ref, u_ref, al_iters=10)
            trajs[-1] = (nominal_states_net, nominal_states, nominal_actions)        
        return trajs, dyn_res

class DEQMPCPolicyHistory(DEQMPCPolicy):
    def __init__(self, args, env):
        super().__init__(args, env)
        self.H = args.H
        if args.deq_out_type == 1:  # deq outputs only state predictions
            self.model = DEQLayerHistoryState(args, env).to(self.device)  
        elif args.deq_out_type == 2:  # deq outputs both state and action predictions
            self.model = DEQLayerHistory(args, env).to(self.device)
        else:
            raise NotImplementedError

    def forward(self, obs_hist, x_gt, u_gt, mask, iter=0, qp_solve=True, lastqp_solve=False):
        """
        Args:
            x_hist (tensor H x bsz x nx): input observation history, including current observation
        """
        if (self.H == 1):
            x_t = obs_hist.reshape(self.bsz, self.nx)
        else:
            x_t = obs_hist[:,-1].reshape(self.bsz, self.nx)
        x_ref = torch.cat([x_t]*self.T, dim=-1).detach().clone()
        x_ref = x_ref.view(-1, self.T, self.nx)
        nominal_actions = torch.zeros((self.bsz, self.T, self.nu), device=self.device)
        z = self.model.init_z(self.bsz).to(self.device)
        out_aux_dict = {"z": z, "x": x_ref, "u": nominal_actions}

        if self.args.solver_type == "al":
            self.tracking_mpc.reinitialize(x_t, mask[:, :, None])

        # run the DEQ layer for deq_iter iterations
        trajs, dyn_res = self.deqmpc_iter(obs_hist, out_aux_dict, x_gt, u_gt, mask, qp_solve, lastqp_solve)        
        return trajs, dyn_res


class DEQMPCPolicyFeedback(DEQMPCPolicy):
    def __init__(self, args, env):
        super().__init__(args, env)
        self.model = DEQLayerFeedback(args, env).to(self.device)

    def forward(self, obs, x_gt, u_gt, mask, iter=0, qp_solve=True, lastqp_solve=False):
        x_ref = torch.cat([obs]*self.T, dim=-1).detach().clone()
        x_ref = x_ref.view(-1, self.T, self.nx)
        nominal_actions = torch.zeros((obs.shape[0], self.T, self.nu), device=self.device)
        z = self.model.init_z(self.bsz).to(self.device)
        # ipdb.set_trace()
        out_aux_dict = {"z": z, "xn": x_ref, "x": x_ref, 'u': nominal_actions}

        if self.args.solver_type == "al":
            self.tracking_mpc.reinitialize(obs, mask[:, :, None])
    
        # run the DEQ layer for deq_iter iterations
        trajs, dyn_res = self.deqmpc_iter(obs, out_aux_dict, x_gt, u_gt, mask, qp_solve, lastqp_solve)        
        return trajs, dyn_res


######################
# Loss computation
######################

def compute_loss_deqmpc(policy, gt_states, gt_actions, gt_mask, trajs):
    return_dict = {"loss": 0.0, "loss_end": 0.0, "losses_var": [], "losses_iter": []}
    loss = 0.0
    # supervise each DEQMPC iteration
    for j, (nominal_states_net, nominal_states, nominal_actions) in enumerate(trajs):
        loss_j = add_loss_based_on_out_type(policy, policy.out_type, policy.loss_type, gt_states,
                                           gt_actions, gt_mask, nominal_states, nominal_actions)
        loss_proxy_j = add_loss_based_on_out_type(policy, policy.out_type, policy.loss_type, gt_states,
                                           gt_actions, gt_mask, nominal_states_net, nominal_actions)
        loss += loss_j + policy.deq_reg * loss_proxy_j
        # return_dict["losses_var"].append(loss_proxy_j.item())
        return_dict["losses_iter"].append(loss_proxy_j.item())
    loss_end = add_loss_based_on_out_type(
        policy, policy.out_type, policy.loss_type, gt_states, gt_actions, gt_mask, nominal_states, nominal_actions)
    return_dict["loss"] = loss
    return_dict["loss_end"] = loss_end
    # ipdb.set_trace()
    return return_dict


def compute_loss_bc(policy, gt_states, gt_actions, gt_mask, trajs):
    return_dict = {"loss": 0.0, "loss_end": 0.0, "losses_var": [], "losses_iter": []}
    loss = 0.0
    nominal_states, nominal_actions = trajs
    loss = add_loss_based_on_out_type(
        policy, policy.out_type, policy.loss_type, gt_states, gt_actions, gt_mask, nominal_states, nominal_actions)
    loss_end = torch.Tensor([0.0])
    return_dict["loss"] = loss
    return_dict["loss_end"] = loss_end
    return return_dict


def add_loss_based_on_out_type(policy, out_type, loss_type, gt_states, gt_actions, gt_mask, nominal_states, nominal_actions):
    loss = 0.0
    if out_type == 0 or out_type == 2:
        # supervise action
        if loss_type == "l2":
            loss += torch.norm((nominal_actions - gt_actions) *
                               gt_mask[:, :, None]).pow(2).mean()
        elif loss_type == "l1":
            loss += torch.abs((nominal_actions - gt_actions) *
                              gt_mask[:, :, None])[:,:policy.T-1].sum(dim=-1).mean()
        # loss += torch.abs((nominal_actions - gt_actions) *
        #                   gt_mask[:, :, None])[:,:policy.T-1].sum(dim=-1).mean()
        # ipdb.set_trace()
    if out_type == 1 or out_type == 2:
        # supervise state
        if loss_type == "l2":
            loss += torch.norm((nominal_states - gt_states) *
                               gt_mask[:, :, None]).pow(2).mean()
        elif loss_type == "l1":
            loss += torch.abs((nominal_states - gt_states) *
                            gt_mask[:, :, None]).sum(dim=-1).mean()
    if out_type == 3:
        # supervise configuration
        if loss_type == "l2":
            loss += torch.norm((nominal_states[..., :policy.nq] - gt_states[..., :policy.nq]) *
                               gt_mask[:, :, None]).pow(2).mean()
        elif loss_type == "l1":
            loss += torch.abs((nominal_states[..., :policy.nq] - gt_states[..., :policy.nq]) *
                          gt_mask[:, :, None]).sum(dim=-1).mean()
        
    return loss


def compute_loss(policy, gt_states, gt_actions, gt_mask, trajs, deq, deqmpc):
    if deq:
        # deq or deqmpc
        if deqmpc:
            # full deqmpc
            return compute_loss_deqmpc(policy, gt_states, gt_actions, gt_mask, trajs)
        else:
            # deq -- pretrain
            return compute_loss_deqmpc(policy.model, gt_states, gt_actions, gt_mask, trajs)
    else:
        # vanilla behavior cloning
        return compute_loss_bc(policy, gt_states, gt_actions, gt_mask, trajs)


######################
# Other policies
######################

class FFDNetwork(torch.nn.Module):
    """
    Feedforward network to generate reference trajectories of horizon T
    """

    def __init__(self, args, env):
        super().__init__()
        self.args = args
        self.nu = env.nu
        self.nx = env.nx
        self.nq = args.nq
        self.T = args.T

        # define the network layers :
        self.fc1 = torch.nn.Linear(self.nx, 256)
        self.ln1 = torch.nn.LayerNorm(256)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(256, 256)
        self.ln2 = torch.nn.LayerNorm(256)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(256, self.nq * self.T)
        self.net = torch.nn.Sequential(
            self.fc1, self.ln1, self.relu1, self.fc2, self.ln2, self.relu2, self.fc3
        )

    def forward(self, x):
        """
        compute the policy output for the given state x
        """
        dx_ref = self.net(x)
        dx_ref = dx_ref.view(-1, self.T, self.nq)
        x_ref = dx_ref + x[:, None, : self.nq]
        return x_ref


class Tracking_MPC(torch.nn.Module):
    def __init__(self, args, env):
        super().__init__()
        self.args = args
        self.nu = env.nu
        self.nx = env.nx
        self.nq = env.nq
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
        self.dtype = torch.float64 if args.dtype == "double" else torch.float32
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
                eps=1e-2,
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
                u_init=self.u_init.transpose(0, 1),
                grad_method=ip_mpc.GradMethods.ANALYTIC,
                solver_type="dense",
                single_qp_solve=self.single_qp_solve,
            )

    def forward(self, x0, xu_ref, x_ref, u_ref, al_iters=2):
        """
        compute the mpc output for the given state x and reference x_ref
        """
        if self.args.solver_type == "al":
            xu_ref = torch.cat([x_ref, u_ref], dim=-1)
            if self.x_init is None:
                self.x_init = self.ctrl.x_init = x_ref.detach().clone()
                self.u_init = self.ctrl.u_init = u_ref.detach().clone()

        self.compute_p(xu_ref)
        # ipdb.set_trace()
        if self.args.solver_type == "al":
            self.ctrl.al_iter = al_iters
            cost = al_utils.QuadCost(self.Q, self.p)
        else:
            cost = ip_mpc.QuadCost(self.Q.transpose(
                0, 1), self.p.transpose(0, 1))
            self.ctrl.u_init = self.u_init.transpose(0, 1)
        # ipdb.set_trace()
        state = x0  # .unsqueeze(0).repeat(self.bsz, 1)
        # ipdb.set_trace()
        nominal_states, nominal_actions = self.ctrl(
            state, cost, self.dyn, self.dyn_jac)
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
            self.bsz, self.T, self.nu, dtype=x.dtype, device=x.device)
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
        self.nq = args.nq
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
        # x_ref = x_ref.view(-1, self.nq)
        x_ref = torch.cat([x_ref, torch.zeros(list(
            x_ref.shape[:-1]) + [self.nq,]).to(self.args.device),], dim=-1,).transpose(0, 1)
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
        self.nq = args.nq
        self.T = args.T
        self.dt = env.dt
        self.device = args.device
        self.hdim = args.hdim
        self.out_type = args.policy_out_type

        # define the network layers :
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.nx, self.hdim),
            torch.nn.LayerNorm(self.hdim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hdim, self.hdim),
            torch.nn.LayerNorm(self.hdim),
            torch.nn.ReLU(),
        )
        if self.out_type == 0:
            self.out_dim = self.nu * self.T
        elif self.out_type == 1:
            self.out_dim = self.nx * self.T
        elif self.out_type == 2:
            self.out_dim = (self.nx + self.nu) * self.T
        elif self.out_type == 3:
            self.out_dim = (self.nq) * self.T

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
        if self.out_type == 0:
            actions = self.model(x)
            actions = actions.view(-1, self.T, self.nu)
            states = None
        elif self.out_type == 1:
            states = self.model(x)
            states = states.view(-1, self.T, self.nx)
            actions = None
        elif self.out_type == 2:
            states = self.model(x)[:, : self.nx * self.T]
            states = states.view(-1, self.T, self.nx)
            actions = self.model(x)[:, self.nx * self.T:]
            actions = actions.view(-1, self.T, self.nu)
        elif self.out_type == 3:
            pos = self.model(x)
            vel = (pos[:, 1:] - pos[:, :-1]) / self.dt
            vel = torch.cat([vel, vel[:, -1:]], dim=1)
            states = torch.cat([pos, vel], dim=-1).view(-1, self.T, self.nx)
            actions = None
        return states, actions

