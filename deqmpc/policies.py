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
        # self.model = DEQLayerDelta(args, env)
        self.model.to(self.device)
        self.out_type = args.policy_out_type  # output type of policy
        self.loss_type = args.loss_type  # loss type for policy
        self.tracking_mpc = Tracking_MPC(args, env)
        self.mpc_time = []
        self.network_time = []

    def forward(self, x, x_gt, u_gt, mask, out_iter=0, qp_solve=True, lastqp_solve=False):
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
        self.x_init = x_ref
        nominal_actions = torch.zeros((x.shape[0], self.T, self.nu), device=self.device)

        z = self.model.init_z(x.shape[0])

        out_aux_dict = {"z": z, "x": x_ref, 'u': nominal_actions}

        if self.args.solver_type == "al":
            self.tracking_mpc.reinitialize(x, mask[:, :, None])
    
        # run the DEQ layer for deq_iter iterations

        policy_out = self.deqmpc_iter(x, out_aux_dict, x_gt, u_gt, mask, qp_solve, lastqp_solve, out_iter)        
        policy_out["init_states"] = x_ref
        return policy_out

    def deqmpc_iter(self, obs, out_aux_dict, x_gt, u_gt, mask, qp_solve=False, lastqp_solve=False, out_iter=0): 
        deq_iter = self.deq_iter   
        opt_from_iter = 0

        trajs = []
        scales = []
        for i in range(deq_iter):
            in_obs_dict = {"o": obs}
            out_aux_dict["iter"] = i
            out_mpc_dict, out_aux_dict = self.model(in_obs_dict, out_aux_dict)
            x_t, x_ref, u_ref = out_mpc_dict["x_t"], out_mpc_dict["x_ref"], out_mpc_dict["u_ref"]
            # if (out_iter == 4900 or out_iter == 5000 or out_iter == 5500):
            # ipdb.set_trace()
            # x_ref[:, 1:, :3] = x_gt[:, 1:, :3]
            # x_ref[:, 1:, 6] = x_gt[:, 1:, 6]
            # x_ref[:, 0] = x_gt[:, 0]
            xu_ref = torch.cat([x_ref, u_ref], dim=-1)
            nominal_states_net = x_ref
            nominal_states = nominal_states_net
            nominal_actions = u_ref

            # Only run MPC after a few iterations, don't flow MPC gradients through the DEQ
            if qp_solve and i >= opt_from_iter:
                nominal_states, nominal_actions = self.tracking_mpc(x_t, xu_ref, x_ref, u_ref, al_iters=2)
                out_aux_dict["x"] = nominal_states#.detach().clone()
                out_aux_dict["u"] = nominal_actions#.detach().clone()
                # out_aux_dict["xn"] = out_aux_dict["xn"].detach().clone()
                # if (out_iter == 5000 or out_iter == 5500):
                #     ipdb.set_trace()
            # if not lastqp_solve:
            #     out_aux_dict["x"] = out_aux_dict["x"].detach().clone()
            #     out_aux_dict["u"] = out_aux_dict["u"].detach().clone()
            # ipdb.set_trace()
            if (qp_solve and i < opt_from_iter) or lastqp_solve:
                trajs.append((nominal_states_net, nominal_states.detach().clone(), nominal_actions.detach().clone()))
            else:
                # scales.append(out_mpc_dict["s"].detach().clone().mean().item())
                # Only supervise DEQ training or joint iterations for DEQMPC
                trajs.append((nominal_states_net, nominal_states, nominal_actions))

        dyn_res = (self.tracking_mpc.dyn(x_gt[:, :-1].reshape(-1, self.nx).double(
        ), u_gt[:, :-1].reshape(-1, self.nu).double()) - x_gt[:,1:].reshape(-1, self.nx)).reshape(self.bsz, -1).norm(dim=1).mean().item()
        ipdb.set_trace()
        self.network_time = []
        self.mpc_time = []

        if lastqp_solve:
            nominal_states, nominal_actions = self.tracking_mpc(
                x_t, xu_ref, x_ref, u_ref, al_iters=10)
            trajs[-1] = (nominal_states_net, nominal_states, nominal_actions)        
        policy_out = {"trajs": trajs, "dyn_res": dyn_res, "scales": scales}    
        return policy_out

class DEQMPCPolicyEE(DEQMPCPolicy):
    def __init__(self, args, env):
        super().__init__(args, env)
        self.model = DEQLayerEE(args, env).to(self.device)
    
    def forward(self, x, x_gt, u_gt, mask, out_iter=0, qp_solve=True, lastqp_solve=False):
        # initialize trajectory with current state
        x_ref = torch.cat([x]*self.T, dim=-1).detach().clone()
        x_ref = x_ref.view(-1, self.T, self.nx)
        self.x_init = x_ref
        nominal_actions = torch.zeros((x.shape[0], self.T, self.nu), device=self.device)

        z = self.model.init_z(x.shape[0])

        out_aux_dict = {"z": z, "x": x_ref, 'u': nominal_actions}

        if self.args.solver_type == "al":
            self.tracking_mpc.reinitialize(x, mask[:, :, None])
    
        # run the DEQ layer for deq_iter iterations

        policy_out = self.deqmpc_iter(x, out_aux_dict, x_gt, u_gt, mask, qp_solve, lastqp_solve, out_iter)        
        policy_out["init_states"] = x_ref
        return policy_out

class DEQMPCPolicyEE2(DEQMPCPolicy):
    def __init__(self, args, env):
        super().__init__(args, env)
        self.model = DEQLayerEE2(args, env).to(self.device)

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

    def forward(self, obs_hist, x_gt, u_gt, mask, out_iter=0, qp_solve=True, lastqp_solve=False):
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
        self.x_init = x_ref
        nominal_actions = torch.zeros((self.bsz, self.T, self.nu), device=self.device)
        z = self.model.init_z(self.bsz)
        out_aux_dict = {"z": z, "x": x_ref, "u": nominal_actions}

        if self.args.solver_type == "al":
            self.tracking_mpc.reinitialize(x_t, mask[:, :, None])

        # run the DEQ layer for deq_iter iterations
        policy_out = self.deqmpc_iter(obs_hist, out_aux_dict, x_gt, u_gt, mask, qp_solve, lastqp_solve)        
        return policy_out

class DEQMPCPolicyHistoryEstPred(DEQMPCPolicy):
    def __init__(self, args, env):
        super().__init__(args, env)
        self.H = args.H
        self.state_estimator = Tracking_MPC(args, env, state_estimator=True)
        if args.deq_out_type == 1:  # deq outputs only state predictions
            self.model = DEQLayerHistoryStateEstPred(args, env).to(self.device)  
        elif args.deq_out_type == 2:  # deq outputs both state and action predictions
            self.model = DEQLayerHistory(args, env).to(self.device)
        else:
            raise NotImplementedError

    def forward(self, obs_hist, x_gt, u_gt, u_gt_est, mask, out_iter=0, qp_solve=True, lastqp_solve=False):
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
        self.x_init = x_ref
        nominal_actions = torch.zeros((self.bsz, self.T, self.nu), device=self.device)
        z = self.model.init_z(self.bsz)
        out_aux_dict = {"z": z, "x": x_ref, "u": nominal_actions, "x_est": obs_hist.reshape(self.bsz, self.H, self.nx)}

        if self.args.solver_type == "al":
            self.tracking_mpc.reinitialize(x_t, mask[:, :, None])
            self.state_estimator.reinitialize(x_t, mask[:, :, None])

        # run the DEQ layer for deq_iter iterations
        policy_out = self.deqmpc_iter(obs_hist, out_aux_dict, x_gt, u_gt, u_gt_est, mask, qp_solve, lastqp_solve)        
        return policy_out

    def deqmpc_iter(self, obs, out_aux_dict, x_gt, u_gt, u_gt_est, mask, qp_solve=False, lastqp_solve=False, out_iter=0): 
        deq_iter = self.deq_iter   
        opt_from_iter = 0

        trajs = []
        scales = []
        nominal_x_ests = []
        for i in range(deq_iter):
            in_obs_dict = {"o": obs}
            out_aux_dict["iter"] = i
            out_mpc_dict, out_aux_dict = self.model(in_obs_dict, out_aux_dict)
            x_t, x_ref, u_ref = out_mpc_dict["x_t"], out_mpc_dict["x_ref"], out_mpc_dict["u_ref"]
            x_est = out_aux_dict["x_est"]
            # if (out_iter == 4900 or out_iter == 5000 or out_iter == 5500):
            #     ipdb.set_trace()
            xu_ref = torch.cat([x_ref, u_ref], dim=-1)
            nominal_states_net = x_ref
            nominal_states = nominal_states_net
            nominal_actions = u_ref
            nominal_x_est = x_est
            xu_est = torch.cat([x_est, u_gt_est], dim=-1)
            x_t_est = x_est[:, 0]

            # Only run MPC after a few iterations, don't flow MPC gradients through the DEQ
            if qp_solve and i >= opt_from_iter:
                nominal_states_est, _ = self.state_estimator(x_t_est, xu_est, x_est, u_gt_est, al_iters=2)
                nominal_states, nominal_actions = self.tracking_mpc(x_t, xu_ref, x_ref, u_ref, al_iters=2)
                out_aux_dict["x"] = nominal_states#.detach().clone()
                out_aux_dict["u"] = nominal_actions#.detach().clone()
                out_aux_dict["x_est"] = nominal_states_est#.detach().clone()
                # out_aux_dict["xn"] = out_aux_dict["xn"].detach().clone()
                # if (out_iter == 5000 or out_iter == 5500):
                #     ipdb.set_trace()
            # if not lastqp_solve:
            #     out_aux_dict["x"] = out_aux_dict["x"].detach().clone()
            #     out_aux_dict["u"] = out_aux_dict["u"].detach().clone()
            nominal_x_ests.append((nominal_x_est, x_est))
            if (qp_solve and i < opt_from_iter) or lastqp_solve:
                trajs.append((nominal_states_net, nominal_states.detach().clone(), nominal_actions.detach().clone()))
            else:
                # scales.append(out_mpc_dict["s"].detach().clone().mean().item())
                # Only supervise DEQ training or joint iterations for DEQMPC
                trajs.append((nominal_states_net, nominal_states, nominal_actions))

        dyn_res = (self.tracking_mpc.dyn(x_gt[:, :-1].reshape(-1, self.nx).double(
        ), u_gt[:, :-1].reshape(-1, self.nu).double()) - x_gt[:,1:].reshape(-1, self.nx)).reshape(self.bsz, -1).norm(dim=1).mean().item()
        self.network_time = []
        self.mpc_time = []

        if lastqp_solve:
            nominal_states, nominal_actions = self.tracking_mpc(
                x_t, xu_ref, x_ref, u_ref, al_iters=10)
            trajs[-1] = (nominal_states_net, nominal_states, nominal_actions)        
        policy_out = {"trajs": trajs, "dyn_res": dyn_res, "scales": scales, "nominal_x_ests": nominal_x_ests}    
        return policy_out

class DEQMPCPolicyFeedback(DEQMPCPolicy):
    def __init__(self, args, env):
        super().__init__(args, env)
        self.model = DEQLayerFeedback(args, env).to(self.device)

    def forward(self, obs, x_gt, u_gt, mask, out_iter=0, qp_solve=True, lastqp_solve=False):
        x_ref = torch.cat([obs]*self.T, dim=-1).detach().clone()
        x_ref = x_ref.view(-1, self.T, self.nx)
        self.x_init = x_ref
        nominal_actions = torch.zeros((obs.shape[0], self.T, self.nu), device=self.device)
        z = self.model.init_z(self.bsz)
        # ipdb.set_trace()
        out_aux_dict = {"z": z, "xn": x_ref, "x": x_ref, 'u': nominal_actions}

        if self.args.solver_type == "al":
            self.tracking_mpc.reinitialize(obs, mask[:, :, None])
    
        # run the DEQ layer for deq_iter iterations
        trajs, dyn_res, scales = self.deqmpc_iter(obs, out_aux_dict, x_gt, u_gt, mask, qp_solve, lastqp_solve, out_iter)        
        return trajs, dyn_res, scales, x_ref

class DEQMPCPolicyQ(DEQMPCPolicy):
    def __init__(self, args, env):
        super().__init__(args, env)
        self.model = DEQLayerQ(args, env).to(self.device)

    def forward(self, obs, x_gt, u_gt, mask, out_iter=0, qp_solve=True, lastqp_solve=False):
        # obs is x0
        x_ref = torch.cat([obs]*self.T, dim=-1).detach().clone()
        x_ref = x_ref.view(-1, self.T, self.nx)
        self.x_init = x_ref
        nominal_actions = torch.zeros((obs.shape[0], self.T, self.nu), device=self.device)
        z = self.model.init_z(self.bsz)
        q = torch.ones_like(x_ref[:,:,0])
        # ipdb.set_trace()
        out_aux_dict = {"z": z, "x": x_ref, 'u': nominal_actions, 'q': q}

        if self.args.solver_type == "al":
            self.tracking_mpc.reinitialize(obs, mask[:, :, None])
    
        # run the DEQ layer for deq_iter iterations
        policy_out = self.deqmpc_iter(obs, out_aux_dict, x_gt, u_gt, mask, qp_solve, lastqp_solve, out_iter)        
        return policy_out

    def deqmpc_iter(self, obs, out_aux_dict, x_gt, u_gt, mask, qp_solve=False, lastqp_solve=False, out_iter=0): 
        deq_iter = self.deq_iter   
        opt_from_iter = 0

        trajs = []
        q_scalings = []
        for i in range(deq_iter):
            in_obs_dict = {"o": obs}
            out_aux_dict["iter"] = i
            out_mpc_dict, out_aux_dict = self.model(in_obs_dict, out_aux_dict)
            x_t, x_ref, u_ref, q_scaling = out_mpc_dict["x_t"], out_mpc_dict["x_ref"], out_mpc_dict["u_ref"], out_mpc_dict["q"]
            # if (out_iter == 4900 or out_iter == 5000 or out_iter == 5500):
            #     ipdb.set_trace()
            xu_ref = torch.cat([x_ref, u_ref], dim=-1)
            ipdb.set_trace()
            nominal_states_net = x_ref
            nominal_states = nominal_states_net
            nominal_actions = u_ref

            # Only run MPC after a few iterations, don't flow MPC gradients through the DEQ
            if qp_solve and i >= opt_from_iter:
                # ipdb.set_trace()
                nominal_states, nominal_actions = self.tracking_mpc(x_t, xu_ref, x_ref, u_ref, q_scaling, al_iters=2)
                # out_aux_dict["x"] = nominal_states.detach().clone()
                # out_aux_dict["u"] = nominal_actions.detach().clone()

                # if (out_iter == 5000 or out_iter == 5500):
                #     ipdb.set_trace()
                
            # if not lastqp_solve:
            #     out_aux_dict["x"] = out_aux_dict["x"].detach().clone()
            #     out_aux_dict["u"] = out_aux_dict["u"].detach().clone()
            #     out_aux_dict["q"] = out_aux_dict["q"].detach().clone()
            
            q_scalings.append(q_scaling)
            if (qp_solve and i < opt_from_iter) or lastqp_solve:
                trajs.append((nominal_states_net, nominal_states.detach().clone(), nominal_actions.detach().clone()))
            else:
                # Only supervise DEQ training or joint iterations for DEQMPC
                trajs.append((nominal_states_net, nominal_states, nominal_actions))

        dyn_res = (self.tracking_mpc.dyn(x_gt[:, :-1].reshape(-1, self.nx).double(
        ), u_gt[:, :-1].reshape(-1, self.nu).double()) - x_gt[:,1:].reshape(-1, self.nx)).reshape(self.bsz, -1).norm(dim=1).mean().item()
        self.network_time = []
        self.mpc_time = []

        if lastqp_solve:
            nominal_states, nominal_actions = self.tracking_mpc(
                x_t, xu_ref, x_ref, u_ref, al_iters=10)
            trajs[-1] = (nominal_states_net, nominal_states, nominal_actions)    

        policy_out = {"trajs": trajs, "dyn_res": dyn_res, "q_scaling": q_scalings}    
        return policy_out


######################
# Loss computation
######################

def compute_loss_deqmpc_invres_l2_old(policy, gt_states, gt_actions, gt_mask, trajs, coeffs=None):
    return_dict = {"loss": 0.0, "loss_end": 0.0, "losses_var": [], "losses_iter": []}
    loss = 0.0
    losses = []
    residuals = []
    lossjs = []
    loss_proxies = []
    if coeffs is None:
        coeffs_pos = coeffs_vel = coeffs_act = torch.ones((len(trajs)), device=gt_states.device)
    else:
        coeffs = coeffs.view(len(trajs), -1)
        coeffs_pos = coeffs[:, 0]
        if coeffs.shape[1] > 1:
            coeffs_vel = coeffs[:, 1]
        else:
            coeffs_vel = torch.ones((len(trajs)), device=gt_states.device)
        if coeffs.shape[1] > 2:
            coeffs_act = coeffs[:, 2]
        else:
            coeffs_act = torch.ones((len(trajs)), device=gt_states.device)
    # supervise each DEQMPC iteration
    for j, (nominal_states_net, nominal_states, nominal_actions) in enumerate(trajs):
        loss_j, res = compute_cost_coeff(policy, policy.out_type, policy.loss_type, gt_states,
                                    gt_actions, gt_mask, nominal_states, nominal_actions,
                                    coeffs_pos[j], coeffs_vel[j], coeffs_act[j])
        loss_proxy_j, res_proxy = compute_cost_coeff(policy, policy.out_type, policy.loss_type, gt_states,
                                    gt_actions, gt_mask, nominal_states_net, nominal_actions,
                                    coeffs_pos[j], coeffs_vel[j], coeffs_act[j])
        losses += [loss_j + policy.deq_reg * loss_proxy_j]
        loss_proxies += [loss_proxy_j]
        lossjs += [loss_j]
        # return_dict["losses_var"].append(loss_proxy_j.item())
        return_dict["losses_iter"].append(loss_proxy_j.mean().item())
        residuals.append(res)
    # ipdb.set_trace()
    residuals = torch.stack(residuals, dim=1)
    inv_residuals = 1/(residuals + 1e-8)
    inv_residuals = inv_residuals / inv_residuals.mean(dim=1, keepdim=True)
    losses = torch.stack(losses, dim=1)*(inv_residuals.detach().clone())
    loss = losses.mean(dim=0).sum()
    loss_end = compute_cost_coeff(
        policy, policy.out_type, policy.loss_type, gt_states, gt_actions, gt_mask,
        nominal_states, nominal_actions, coeffs_pos[-1], coeffs_vel[-1], coeffs_act[-1])[0].mean()
    return_dict["loss"] = loss
    return_dict["loss_end"] = loss_end
    # ipdb.set_trace()
    return return_dict

def compute_loss_deqmpc(policy, gt_states, gt_actions, gt_mask, policy_out, coeffs=None):
    return_dict = {"loss": 0.0, "loss_end": 0.0, "losses_var": [], "losses_iter_opt": [], "losses_iter_nn": [], "losses_iter_base": [], "losses_iter": []}
    trajs = policy_out["trajs"]
    loss = 0.0
    losses = []
    residuals = []
    loss_opts = []
    loss_nns = []
    if coeffs is None:
        coeffs_pos = coeffs_vel = coeffs_act = torch.ones((len(trajs)), device=gt_states.device)
    else:
        coeffs = coeffs.view(len(trajs), -1)
        coeffs_pos = coeffs[:, 0]
        if coeffs.shape[1] > 1:
            coeffs_vel = coeffs[:, 1]
        else:
            coeffs_vel = torch.ones((len(trajs)), device=gt_states.device)
        if coeffs.shape[1] > 2:
            coeffs_act = coeffs[:, 2]
        else:
            coeffs_act = torch.ones((len(trajs)), device=gt_states.device)
    # supervise each DEQMPC iteration # replace x_init with gt[0]
    loss_init, res_init = compute_cost_coeff(policy, policy.out_type, policy.loss_type, gt_states,
                                    gt_actions, gt_mask, policy.x_init, trajs[0][-1]*0,
                                    coeffs_pos[0], coeffs_vel[0], coeffs_act[0])
    residuals.append(res_init)
    for j, (nominal_states_net, nominal_states, nominal_actions) in enumerate(trajs):
        loss_opt_j, res = compute_cost_coeff(policy, policy.out_type, policy.loss_type, gt_states,
                                    gt_actions, gt_mask, nominal_states, nominal_actions,
                                    coeffs_pos[j], coeffs_vel[j], coeffs_act[j])
        loss_nn_j, res_nn = compute_cost_coeff(policy, policy.out_type, policy.loss_type, gt_states,
                                    gt_actions, gt_mask, nominal_states_net, nominal_actions,
                                    coeffs_pos[j], coeffs_vel[j], coeffs_act[j])
        losses += [loss_opt_j + policy.deq_reg * loss_nn_j]
        loss_nns += [loss_nn_j]
        loss_opts += [loss_opt_j]
        # return_dict["losses_var"].append(loss_proxy_j.item())
        # ipdb.set_trace()
        return_dict["losses_iter_opt"].append(loss_opt_j.mean().item())
        return_dict["losses_iter_nn"].append(loss_nn_j.mean().item())
        return_dict["losses_iter_base"].append(losses[-1].mean().item())
        return_dict["losses_iter"].append(losses[-1].mean().item())
        residuals.append(res)
    ### compute iteration weights based on previous losses and compute example weights based on net residuals
    ### iteration weights
    residuals = torch.stack(residuals, dim=1)
    weight_mask = gt_mask.sum(dim=1) == 1
    iter_weights = 5**(torch.log(residuals[:,:1]/(10*residuals[:,:-1])))
    iter_weights[weight_mask] = 1
    iter_weights = iter_weights / iter_weights.sum(dim=1, keepdim=True)
    ex_weights = residuals.mean(dim=1, keepdim=True)#**2
    ex_weights = ex_weights / ex_weights.mean()
    losses = torch.stack(losses, dim=1)#*(ex_weights.detach().clone())#*(iter_weights.detach().clone())
    loss = losses.mean(dim=0).sum()
    loss_end = compute_cost_coeff(
        policy, policy.out_type, policy.loss_type, gt_states, gt_actions, gt_mask,
        nominal_states, nominal_actions, coeffs_pos[-1], coeffs_vel[-1], coeffs_act[-1])[0].mean()
    return_dict["loss"] = loss
    return_dict["loss_end"] = loss_end
    return_dict["ex_weights"] = ex_weights
    return_dict["iter_weights"] = iter_weights
    return return_dict

def compute_loss_deqmpc_hist(policy, gt_states, gt_actions, gt_obs, gt_mask, policy_out, coeffs=None):
    return_dict = {"loss": 0.0, "loss_end": 0.0, "losses_var": [], "losses_iter_opt": [], "losses_iter_nn": [], "losses_iter_base": [], "losses_iter": [], "losses_x_ests": []}
    trajs = policy_out["trajs"]
    loss = 0.0
    losses = []
    residuals = []
    loss_opts = []
    loss_nns = []
    if coeffs is None:
        coeffs_pos = coeffs_vel = coeffs_act = torch.ones((len(trajs)), device=gt_states.device)
    else:
        coeffs = coeffs.view(len(trajs), -1)
        coeffs_pos = coeffs[:, 0]
        if coeffs.shape[1] > 1:
            coeffs_vel = coeffs[:, 1]
        else:
            coeffs_vel = torch.ones((len(trajs)), device=gt_states.device)
        if coeffs.shape[1] > 2:
            coeffs_act = coeffs[:, 2]
        else:
            coeffs_act = torch.ones((len(trajs)), device=gt_states.device)
    # supervise each DEQMPC iteration # replace x_init with gt[0]
    loss_init, res_init = compute_cost_coeff(policy, policy.out_type, policy.loss_type, gt_states,
                                    gt_actions, gt_mask, policy.x_init, trajs[0][-1]*0,
                                    coeffs_pos[0], coeffs_vel[0], coeffs_act[0])
    residuals.append(res_init)
    for j, (nominal_states_net, nominal_states, nominal_actions) in enumerate(trajs):
        loss_opt_j, res = compute_cost_coeff(policy, policy.out_type, policy.loss_type, gt_states,
                                    gt_actions, gt_mask, nominal_states, nominal_actions,
                                    coeffs_pos[j], coeffs_vel[j], coeffs_act[j])
        loss_nn_j, res_nn = compute_cost_coeff(policy, policy.out_type, policy.loss_type, gt_states,
                                    gt_actions, gt_mask, nominal_states_net, nominal_actions,
                                    coeffs_pos[j], coeffs_vel[j], coeffs_act[j])
        loss_hist_j, res_hist = compute_cost_coeff(policy, policy.out_type, policy.loss_type, gt_obs,
                                    gt_actions, gt_mask*0+1, policy_out["nominal_x_ests"][j][0], nominal_actions,
                                    coeffs_pos[j], coeffs_vel[j], coeffs_act[j])
        loss_hist_nn_j, res_hist = compute_cost_coeff(policy, policy.out_type, policy.loss_type, gt_obs,
                                    gt_actions, gt_mask*0+1, policy_out["nominal_x_ests"][j][1], nominal_actions,
                                    coeffs_pos[j], coeffs_vel[j], coeffs_act[j])
        losses += [loss_opt_j + policy.deq_reg * loss_nn_j]# + loss_hist_j + policy.deq_reg * loss_hist_nn_j]
        loss_nns += [loss_nn_j]
        loss_opts += [loss_opt_j]
        # return_dict["losses_var"].append(loss_proxy_j.item())
        return_dict["losses_iter_opt"].append(loss_opt_j.mean().item())
        return_dict["losses_iter_nn"].append(loss_nn_j.mean().item())
        return_dict["losses_iter_base"].append(losses[-1].mean().item())
        return_dict["losses_iter"].append(losses[-1].mean().item())
        return_dict["losses_x_ests"].append(loss_hist_j.mean().item())
        residuals.append(res)
    ### compute iteration weights based on previous losses and compute example weights based on net residuals
    ### iteration weights
    residuals = torch.stack(residuals, dim=1)
    weight_mask = gt_mask.sum(dim=1) == 1
    iter_weights = 5**(torch.log(residuals[:,:1]/(10*residuals[:,:-1])))
    iter_weights[weight_mask] = 1
    iter_weights = iter_weights / iter_weights.sum(dim=1, keepdim=True)
    ex_weights = residuals.mean(dim=1, keepdim=True)#**2
    ex_weights = ex_weights / ex_weights.mean()
    losses = torch.stack(losses, dim=1)#*(ex_weights.detach().clone())#*(iter_weights.detach().clone())
    loss = losses.mean(dim=0).sum()
    loss_end = compute_cost_coeff(
        policy, policy.out_type, policy.loss_type, gt_states, gt_actions, gt_mask,
        nominal_states, nominal_actions, coeffs_pos[-1], coeffs_vel[-1], coeffs_act[-1])[0].mean()
    return_dict["loss"] = loss
    return_dict["loss_end"] = loss_end
    return_dict["ex_weights"] = ex_weights
    return_dict["iter_weights"] = iter_weights
    return return_dict


def compute_gradratios_deqmpc(policy, gt_states, gt_actions, gt_mask, policy_out):
    trajs = policy_out["trajs"]
    return_dict = {"loss": 0.0, "loss_end": 0.0, "losses_var": [], "losses_iter": []}
    losses = []
    loss_proxies = []
    # supervise each DEQMPC iteration
    for j, (nominal_states_net, nominal_states, nominal_actions) in enumerate(trajs):
        loss_j = compute_decomposed_loss(policy, policy.out_type, policy.loss_type, gt_states,
                                           gt_actions, gt_mask, nominal_states, nominal_actions)
        loss_proxy_j = compute_decomposed_loss(policy, policy.out_type, policy.loss_type, gt_states,
                                           gt_actions, gt_mask, nominal_states_net, nominal_actions)
        loss_proxies += loss_proxy_j
        losses += loss_j #+ policy.deq_reg * loss_proxy_j
        # return_dict["losses_var"].append(loss_proxy_j.item())
        # return_dict["losses_iter"].append(loss_proxy_j.item())
    losses = torch.stack(losses, dim=0)
    loss_proxies = torch.stack(loss_proxies, dim=0)
    
    grads = torch.stack([torch.autograd.grad(losses[i] + policy.deq_reg*loss_proxies[i], policy.model.out_layer[0].weight, retain_graph=True)[0].view(-1) for i in range(len(losses))], dim=0).norm(dim=-1)
    grad_idx = torch.where(grads>1e-8)[0][0]
    grad_ratios = grads[grad_idx] / grads
    grad_ratios = torch.where(grad_ratios > 1e6, torch.ones_like(grad_ratios), grad_ratios)
    # compute moving averages
    return grad_ratios, losses.reshape((len(trajs), len(loss_j))).mean(dim=-1), loss_proxies.reshape((len(trajs), len(loss_j))).mean(dim=-1)

def compute_loss_deqmpc_qscaling(policy, gt_states, gt_actions, gt_mask, policy_out, coeffs=None):
    return_dict = {"loss": 0.0, "loss_end": 0.0, "losses_var": [], "losses_iter_opt": [], "losses_iter_nn": [], "losses_iter_base": [], "losses_iter": [], "q_scaling": []}
    trajs = policy_out["trajs"]
    q_scaling = policy_out["q_scaling"]
    loss = 0.0
    # supervise each DEQMPC iteration
    losses = []
    residuals = []
    loss_opts = []
    loss_nns = []
    if coeffs is None:
        coeffs_pos = coeffs_vel = coeffs_act = torch.ones((len(trajs)), device=gt_states.device)
    else:
        coeffs = coeffs.view(len(trajs), -1)
        coeffs_pos = coeffs[:, 0]
        if coeffs.shape[1] > 1:
            coeffs_vel = coeffs[:, 1]
        else:
            coeffs_vel = torch.ones((len(trajs)), device=gt_states.device)
        if coeffs.shape[1] > 2:
            coeffs_act = coeffs[:, 2]
        else:
            coeffs_act = torch.ones((len(trajs)), device=gt_states.device)
    # supervise each DEQMPC iteration
    loss_init, res_init = compute_cost_coeff(policy, policy.out_type, policy.loss_type, gt_states,
                                    gt_actions, gt_mask, policy.x_init, trajs[0][-1]*0,
                                    coeffs_pos[0], coeffs_vel[0], coeffs_act[0])
    residuals.append(res_init)
    for j, (nominal_states_net, nominal_states, nominal_actions) in enumerate(trajs):
        loss_opt_j, res = compute_cost_coeff(policy, policy.out_type, policy.loss_type, gt_states,
                                    gt_actions, gt_mask, nominal_states, nominal_actions,
                                    coeffs_pos[j], coeffs_vel[j], coeffs_act[j])
        loss_nn_j, res_nn = compute_cost_coeff(policy, policy.out_type, policy.loss_type, gt_states,
                                    gt_actions, gt_mask, nominal_states_net, nominal_actions,
                                    coeffs_pos[j], coeffs_vel[j], coeffs_act[j])
        q_scaling_j = q_scaling[j]
        loss_q_scaling_j = torch.abs(q_scaling_j - 1.0).sum(dim=1)
        losses += [loss_opt_j + policy.deq_reg * loss_nn_j + 0.02 * loss_q_scaling_j]
        loss_nns += [loss_nn_j]
        loss_opts += [loss_opt_j]
        # return_dict["losses_var"].append(loss_proxy_j.item())
        return_dict["losses_iter_opt"].append(loss_opt_j.mean().item())
        return_dict["losses_iter_nn"].append(loss_nn_j.mean().item())
        return_dict["losses_iter_base"].append((loss_opt_j + policy.deq_reg * loss_nn_j).mean().item())
        return_dict["losses_iter"].append(losses[-1].mean().item())
        return_dict["q_scaling"].append(loss_q_scaling_j.mean().item())
        residuals.append(res)
    ### compute iteration weights based on previous losses and compute example weights based on net residuals
    ### iteration weights
    residuals = torch.stack(residuals, dim=1)
    weight_mask = gt_mask.sum(dim=1) == 1
    iter_weights = 5**(torch.log(residuals[:,:1]/(10*residuals[:,:-1])))
    iter_weights[weight_mask] = 1
    iter_weights = iter_weights / iter_weights.sum(dim=1, keepdim=True)
    ex_weights = residuals.mean(dim=1, keepdim=True)#**2
    ex_weights = ex_weights / ex_weights.mean()
    losses = torch.stack(losses, dim=1)#*(ex_weights.detach().clone())#*(iter_weights.detach().clone())
    loss = losses.mean(dim=0).sum()
    loss_end = compute_cost_coeff(
        policy, policy.out_type, policy.loss_type, gt_states, gt_actions, gt_mask,
        nominal_states, nominal_actions, coeffs_pos[-1], coeffs_vel[-1], coeffs_act[-1])[0].mean()
    return_dict["loss"] = loss
    return_dict["loss_end"] = loss_end
    return_dict["ex_weights"] = ex_weights
    return_dict["iter_weights"] = iter_weights
    return return_dict

def compute_loss_bc(policy, gt_states, gt_actions, gt_mask, policy_out):
    return_dict = {"loss": 0.0, "loss_end": 0.0, "losses_var": [], "losses_iter": []}
    trajs = policy_out["trajs"]
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

def compute_cost_coeff(policy, out_type, loss_type, gt_states, gt_actions, gt_mask, nominal_states, nominal_actions, coeffs_act, coeffs_pos, coeffs_vel):
    loss = 0.0
    resi = resj = resk = 0.0
    if out_type == 0 or out_type == 2:
        # supervise action
        _, lossk, resk = loss_type_conditioned_compute_loss(nominal_actions[:, :policy.T-1], gt_actions[:, :policy.T-1], gt_mask[:, :policy.T-1], loss_type)
        loss = lossk*coeffs_act
    if out_type == 1 or out_type == 2:
        # supervise state
        _, lossi, resi = loss_type_conditioned_compute_loss(nominal_states[:, :, :policy.nq], gt_states[:, :, :policy.nq], gt_mask, loss_type)
        _, lossj, resj = loss_type_conditioned_compute_loss(nominal_states[:, :, policy.nq:], gt_states[:, :, policy.nq:], gt_mask, loss_type)

        # _, lossi, resi = loss_type_conditioned_compute_loss(nominal_states[...,6:7], gt_states[...,6:7], gt_mask, loss_type)
        # _, lossj, resj = loss_type_conditioned_compute_loss(nominal_states[...,13:14], gt_states[...,13:14], gt_mask, loss_type)
        loss += lossi*coeffs_pos + lossj*coeffs_vel
    if out_type == 3:
        # supervise configuration
        _, lossi, resi = loss_type_conditioned_compute_loss(nominal_states[:, :, :policy.nq], gt_states[:, :, :policy.nq], gt_mask, loss_type)
        loss += lossi*coeffs_pos
    return loss, resi + resj + resk

def compute_decomposed_loss(policy, out_type, loss_type, gt_states, gt_actions, gt_mask, nominal_states, nominal_actions):
    loss = []
    if out_type == 0 or out_type == 2:
        # supervise action
        loss += [loss_type_conditioned_compute_loss(nominal_actions[:, :policy.T-1], gt_actions[:, :policy.T-1], gt_mask[:, :policy.T-1], loss_type)[0]]
    if out_type == 1 or out_type == 2:
        # supervise state
        loss += [loss_type_conditioned_compute_loss(nominal_states[:, :, :policy.nq], gt_states[:, :, :policy.nq], gt_mask, loss_type)[0]]
        loss += [loss_type_conditioned_compute_loss(nominal_states[:, :, policy.nq:], gt_states[:, :, policy.nq:], gt_mask, loss_type)[0]]
    if out_type == 3:
        # supervise configuration
        loss += [loss_type_conditioned_compute_loss(nominal_states[:, :, :policy.nq], gt_states[:, :, :policy.nq], gt_mask, loss_type)[0]]
    return loss

def loss_type_conditioned_compute_loss(pred, targ, mask, loss_type):
    res = torch.abs((pred - targ) * mask[:, :, None]).sum(dim=-1)
    if loss_type == "l2":        
        l2 = torch.norm((pred - targ) * mask[:, :, None], dim=-1).pow(2)
        return l2.mean(), l2.mean(dim=1), res.mean(dim=1)
    elif loss_type == "l1":
        l1 = torch.abs((pred - targ) * mask[:, :, None]).sum(dim=-1)
        return l1.mean(), l1.mean(dim=1), res.mean(dim=1)
    elif loss_type == "hinge":
        l1 = torch.abs((pred - targ) * mask[:, :, None])
        l2 = ((pred - targ) * mask[:, :, None]).pow(2)
        hingel = torch.min(l1, l2).sum(dim=-1)
        return hingel.mean(), hingel.mean(dim=1), res.mean(dim=1)

def compute_loss(policy, gt_states, gt_actions, gt_obs, gt_mask, policy_out, deq, deqmpc, coeffs=None):
    if deq:
        # deq or deqmpc
        if deqmpc:
            # full deqmpc
            if "nominal_x_ests" in policy_out.keys():
                return compute_loss_deqmpc_hist(policy, gt_states, gt_actions, gt_obs, gt_mask, policy_out, coeffs=coeffs)
            elif "q_scaling" in policy_out.keys():
                return compute_loss_deqmpc_qscaling(policy, gt_states, gt_actions, gt_mask, policy_out, coeffs=coeffs)
            else:
                return compute_loss_deqmpc(policy, gt_states, gt_actions, gt_mask, policy_out, coeffs=coeffs)
        else:
            # deq -- pretrain
            return compute_loss_deqmpc(policy, gt_states, gt_actions, gt_mask, policy_out, coeffs=coeffs)
    else:
        # vanilla behavior cloning
        return compute_loss_bc(policy, gt_states, gt_actions, gt_mask, policy_out, coeffs=coeffs)
    
def compute_grad_coeff(policy, gt_states, gt_actions, gt_mask, policy_out, deq, deqmpc):
    if deq:
        # deq or deqmpc
        if deqmpc:
            # full deqmpc
            # if "q_scaling" in policy_out.keys():
            #     return compute_gradratios_deqmpc_qscaling(policy, gt_states, gt_actions, gt_mask, policy_out)
            # else:
            return compute_gradratios_deqmpc(policy, gt_states, gt_actions, gt_mask, policy_out)
        else:
            # deq -- pretrain
            return compute_gradratios_deqmpc(policy, gt_states, gt_actions, gt_mask, policy_out)
    else:
        # vanilla behavior cloning
        return compute_loss_bc(policy, gt_states, gt_actions, gt_mask, policy_out)


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
    def __init__(self, args, env, state_estimator=False):
        super().__init__()
        self.args = args
        self.nu = env.nu
        self.nx = env.nx
        self.nq = env.nq
        self.dt = env.dt
        self.T = args.T
        self.dyn = env.dynamics
        self.dyn_jac = env.dynamics_derivatives
        self.state_estimator = state_estimator

        # May comment out input constraints for now
        self.device = args.device
        self.u_upper = torch.tensor(env.action_space.high).to(self.device)
        self.u_lower = torch.tensor(env.action_space.low).to(self.device)
        self.qp_iter = args.qp_iter
        self.eps = args.eps
        self.warm_start = args.warm_start
        self.bsz = args.bsz

        self.Q = args.Q.to(self.device)
        self.Qaux = args.Qaux.to(self.device)
        self.R = args.R.to(self.device)
        self.dtype = torch.float64 if args.dtype == "double" else torch.float32
        # self.Qf = args.Qf
        if args.Q is None:
            self.Q = torch.ones(self.nx, dtype=self.dtype, device=self.device)
            # self.Qf = torch.ones(self.nx, dtype=torch.float32, device=self.device)
            self.R = torch.ones(self.nu, dtype=self.dtype, device=self.device)
        self.Q = torch.cat([self.Q, self.R], dim=0).to(self.dtype)
        self.Qaux = torch.cat([self.Qaux, self.R], dim=0).to(self.dtype)
        self.Q = torch.diag(self.Q).repeat(self.bsz, self.T, 1, 1)
        self.Qaux = torch.diag(self.Qaux).repeat(self.bsz, self.T, 1, 1)
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
                state_estimator=self.state_estimator,
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

    def forward(self, x0, xu_ref, x_ref, u_ref, q_scaling=None, al_iters=2):
        """
        compute the mpc output for the given state x and reference x_ref
        """
        if self.args.solver_type == "al":
            xu_ref = torch.cat([x_ref, u_ref], dim=-1)
            if self.x_init is None:
                self.x_init = self.ctrl.x_init = x_ref.detach().clone()
                self.u_init = self.ctrl.u_init = u_ref.detach().clone()
        if (q_scaling is not None):
            # ipdb.set_trace()
            q_scaling = q_scaling + torch.ones_like(q_scaling)
            Q = self.Q * q_scaling[:,:,None,None]
        else:
            Q = self.Q + self.Qaux
        self.compute_p(xu_ref, Q)
        # ipdb.set_trace()
        if self.args.solver_type == "al":
            self.ctrl.al_iter = al_iters
            cost = al_utils.QuadCost(Q, self.p)
        else:
            cost = ip_mpc.QuadCost(Q.transpose(
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

    def compute_p(self, xu_ref, Q):
        """
        compute the p for the quadratic objective using self.Q as the diagonal matrix and the reference x_ref at each time without a for loop
        """
        # self.p = torch.zeros(
        #     self.T, self.bsz, self.nx + self.nu, dtype=torch.float32, device=self.device
        # )
        # self.p[:, :, : self.nx] = -(
        #     self.Q[:, :, : self.nx, : self.nx] * x_ref.unsqueeze(-2)
        # ).sum(dim=-1)
        targ_pos = torch.zeros(self.nx+self.nu).to(self.device)
        targ_pos[0:3] = torch.tensor([7.4720e-02, -1.3457e-01,  2.4619e-01]).to(self.device)
        targ_pos[6] = torch.pi # upright pendulum
        targ_pos = targ_pos.repeat(self.bsz, self.T, 1)
        self.p = -(Q * xu_ref.unsqueeze(-2)).sum(dim=-1) - (self.Qaux * targ_pos.unsqueeze(-2)).sum(dim=-1)
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

