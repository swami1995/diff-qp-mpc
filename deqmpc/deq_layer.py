import numpy as nq
import torch
import torch.nn as nn
import torch.autograd as autograd
import qpth.al_utils as al_utils
import ipdb
# from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
# from policy_utils import SinusoidalPosEmb
import time

# POSSIBLE OUTPUT TYPES OF DEQ LAYER
# 0: action prediction u[0]->u[T-1]
# 1: state prediction x[0]->x[T-1], would be x[1]->x[T-1] only if using DEQLayer
# 2: state prediction x[0]->x[T-1] and control prediction u[0]->u[T-1]

class DEQLayer(torch.nn.Module):
    '''
    Base class for different DEQ architectures, child classes define `forward`, `setup_input_layer` and `setup_input_layer`
    '''
    def __init__(self, args, env):
        super().__init__()
        self.args = args
        self.nu = env.nu
        self.nx = env.nx
        self.nq = args.nq
        self.dt = env.dt
        self.T = args.T
        self.hdim = args.hdim
        self.layer_type = args.layer_type
        self.inp_type = ""  # args.inp_type
        self.out_type = args.deq_out_type
        self.deq_reg = args.deq_reg
        self.kernel_width = args.kernel_width
        self.pooling = args.pooling
        self.deq_expand = 4
        self.kernel_width_out = 1

        self.setup_input_layer()
        self.setup_deq_layer()
        self.setup_output_layer()

    # TO BE OVERRIDEN
    def forward(self, in_obs_dict, in_aux_dict):
        """
        compute the policy output for the given observation input and feedback input 
        """
        obs, x_prev, z = in_obs_dict["o"], in_aux_dict["x"], in_aux_dict["z"]
        if (x_prev.shape[1] != self.T - 1):  # handle the case of orginal DEQLayer not predicting current state
            x_prev = x_prev[:, 1:]
        bsz = obs.shape[0]
        _obs = obs.reshape(bsz,1,self.nx)
        _input = torch.cat([_obs, x_prev], dim=-2).reshape(bsz, -1)
        _input1 = self.input_layer(_input)
        z_out = self.deq_layer(_input1, z)
        dx_ref = self.output_layer(z_out)

        dx_ref = dx_ref.view(-1, self.T - 1, self.nx)
        vel_ref = dx_ref[..., self.nq:]
        dx_ref = dx_ref[..., :self.nq] * self.dt
        x_ref = torch.cat([dx_ref + x_prev[..., :self.nq], vel_ref], dim=-1)
        x_ref = torch.cat([_obs, x_ref], dim=-2)
        u_ref = torch.zeros_like(x_ref[..., :self.nu])
        
        out_mpc_dict = {"x_t": obs, "x_ref": x_ref, "u_ref": u_ref}
        out_aux_dict = {"x": x_ref[:,1:], "u": u_ref, "z": z_out}
        return out_mpc_dict, out_aux_dict

    def input_layer(self, x):
        if self.layer_type == "mlp":
            inp = self.inp_layer(x)
        elif self.layer_type == "gcn":
            t = self.time_emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
            x = x.reshape(-1, self.T, self.nx)
            x_emb = self.node_encoder(x)
            x0_emb = self.x0_encoder(x[:, 0]).unsqueeze(1).repeat(1, self.T, 1)  #TODO switch case for out_type
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
            out = self.lndeq3(self.reludeq2(
                z + self.lndeq2(x + self.fcdeq2(z))))
        elif self.layer_type == "gcn":
            z = z.view(-1, self.T, self.hdim)
            z = self.convdeq1(z.permute(0, 2, 1))
            z = self.mishdeq1(z)
            z = self.gndeq1(z)
            out = self.gndeq3(self.mishdeq2(
                z + self.gndeq2(x + self.convdeq2(z))))
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

    # TO BE OVERRIDEN
    def setup_input_layer(self):
        self.in_dim = self.nx + self.nx * (self.T - 1) # current state and state prediction
        if self.layer_type == "mlp":
            # ipdb.set_trace()
            self.inp_layer = torch.nn.Sequential(
                torch.nn.Linear(self.in_dim, self.hdim),
                torch.nn.LayerNorm(self.hdim),
                # torch.nn.ReLU()
            )
            # self.fc_inp = torch.nn.Linear(self.nx + self.nq*self.T, self.hdim)
            # self.ln_inp = torch.nn.LayerNorm(self.hdim)
        elif self.layer_type == "gcn":
            # Get sinusoidal embeddings for the time steps
            # self.time_encoder = nn.Sequential(
            #     SinusoidalPosEmb(self.hdim),
            #     nn.Linear(self.hdim, self.hdim*4),
            #     nn.Mish(),
            #     nn.Linear(self.hdim*4, self.hdim),
            #     nn.LayerNorm(self.hdim)
            #     )
            self.time_emb = torch.nn.Parameter(torch.randn(self.T, self.hdim))
            # Get the node embeddings
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
            self.convdeq1 = torch.nn.Conv1d(
                self.hdim, self.hdim*self.deq_expand, self.kernel_width)
            self.convdeq2 = torch.nn.Conv1d(
                self.hdim*self.deq_expand, self.hdim, self.kernel_width)
            self.mishdeq1 = torch.nn.Mish()
            self.mishdeq2 = torch.nn.Mish()
            self.gndeq1 = torch.nn.GroupNorm(
                self.num_groups, self.hdim*self.deq_expand)
            self.gndeq2 = torch.nn.GroupNorm(self.num_groups, self.hdim)
            self.gndeq3 = torch.nn.GroupNorm(self.num_groups, self.hdim)
        elif self.layer_type == "gat":
            NotImplementedError

    # TO BE OVERRIDEN
    def setup_output_layer(self):  
        self.out_dim = self.nx * (self.T-1)  # state prediction
        if self.layer_type == "mlp":
            self.out_layer = torch.nn.Sequential(
                torch.nn.Linear(self.hdim, self.out_dim)
            )
        elif self.layer_type == "gcn":
            self.convout = torch.nn.Conv1d(
                self.hdim, self.hdim, self.kernel_width)
            self.gnout = torch.nn.GroupNorm(self.num_groups, self.hdim)
            self.mishout = torch.nn.Mish()
            self.final_layer = torch.nn.Conv1d(
                self.hdim, self.nq, self.kernel_width_out)
        elif self.layer_type == "gat":
            NotImplementedError

class DEQLayerHistoryState(DEQLayer):
    '''
    DEQ layer takes state history, outputs current state and state prediction
    '''
    def __init__(self, args, env):
        self.H = args.H  # number of history steps (including current state)
        super().__init__(args, env)        
    
    def forward(self, in_obs_dict, in_aux_dict):
        """
        compute the policy output for the given observation input and feedback input 
        """
        obs, x, z = in_obs_dict["o"], in_aux_dict["x"], in_aux_dict["z"]
        bsz = obs.shape[0]
        _obs = obs.reshape(bsz, -1)
        _x = x.reshape(bsz, -1)
        _input = torch.cat([_obs, _x], dim=-1)

        _input1 = self.input_layer(_input)
        z_out = self.deq_layer(_input1, z)
        _dx_ref = self.output_layer(z_out)

        dx_ref = _dx_ref.view(-1, self.T, self.nx)
        vel_ref = dx_ref[..., self.nq:]
        dx_ref = dx_ref[..., :self.nq] * self.dt
        # ipdb.set_trace()
        x_ref = torch.cat([dx_ref + x[..., :self.nq], vel_ref], dim=-1)
        u_ref = torch.zeros_like(x_ref[..., :self.nu])
        
        out_mpc_dict = {"x_t": x_ref[:,0], "x_ref": x_ref, "u_ref": u_ref}
        out_aux_dict = {"x": x_ref, "u": u_ref, "z": z_out}
        return out_mpc_dict, out_aux_dict
    
    def setup_input_layer(self):
        self.in_dim = self.nx * self.H + self.nx * self.T # external input and aux input
        if self.layer_type == "mlp":
            # ipdb.set_trace()
            self.inp_layer = torch.nn.Sequential(
                torch.nn.Linear(self.in_dim, self.hdim),
                torch.nn.LayerNorm(self.hdim),
            )
        else:
            NotImplementedError

    def setup_output_layer(self):  
        self.out_dim = self.nx * self.T
        if self.layer_type == "mlp":
            self.out_layer = torch.nn.Sequential(
                torch.nn.Linear(self.hdim, self.out_dim)
            )
        else:
            NotImplementedError


class DEQLayerHistory(DEQLayer):
    '''
    DEQ layer takes state history, outputs state and action predictions
    '''
    def __init__(self, args, env):
        self.H = args.H  # number of history steps (including current state)
        super().__init__(args, env)        
    
    def forward(self, in_obs_dict, in_aux_dict):
        """
        compute the policy output for the given observation input and feedback input 
        """
        obs, x, u, z = in_obs_dict["o"], in_aux_dict["x"], in_aux_dict["u"], in_aux_dict["z"]
        bsz = obs.shape[0]
        _obs = obs.reshape(bsz, -1)
        _x = x.reshape(bsz, -1)
        _u = u[:,:self.T-1].reshape(bsz, -1)  # remove last action
        _input = torch.cat([_obs, _x, _u], dim=-1)

        _input1 = self.input_layer(_input)
        z_out = self.deq_layer(_input1, z)
        _dxu_ref = self.output_layer(z_out)

        dx_ref = _dxu_ref[..., :self.nx*self.T].reshape(-1, self.T, self.nx)
        u_ref = _dxu_ref[..., self.nx*self.T:].reshape(-1, self.T-1, self.nu)
        u_ref = torch.cat([u_ref, torch.zeros_like(u_ref[:, -1:])], dim=1)  # append zero to last action
        vel_ref = dx_ref[..., self.nq:]
        dx_ref = dx_ref[..., :self.nq] * self.dt
        x_ref = torch.cat([dx_ref + x[..., :self.nq], vel_ref], dim=-1)
        
        out_mpc_dict = {"x_t": x_ref[:,0], "x_ref": x_ref, "u_ref": u_ref}
        out_aux_dict = {"x": x_ref, "u": u_ref, "z": z_out}
        return out_mpc_dict, out_aux_dict
    
    def setup_input_layer(self):
        self.in_dim = self.nx * self.H + self.nx * self.T + self.nu * (self.T-1) # external input and aux input
        if self.layer_type == "mlp":
            # ipdb.set_trace()
            self.inp_layer = torch.nn.Sequential(
                torch.nn.Linear(self.in_dim, self.hdim),
                torch.nn.LayerNorm(self.hdim),
            )
        else:
            NotImplementedError

    def setup_output_layer(self):  
        self.out_dim = self.nx * self.T + self.nu * (self.T-1)
        if self.layer_type == "mlp":
            self.out_layer = torch.nn.Sequential(
                torch.nn.Linear(self.hdim, self.out_dim)
            )
        else:
            NotImplementedError


####################
# End
####################

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

        self.fc_out = torch.nn.Linear(self.hdim, self.nx * self.T)

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
        x_ref = x_ref.view(-1, self.T, self.nx)
        x_ref = x_ref + x[:, None, : self.nx] * 10
        return x_ref

    def deq_fixed_point(self, x, z):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z, self.forward_res = self.solver(
                lambda z: self.f(z, x), z, **self.kwargs)
        z = self.f(z, x)

        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, x)

        def backward_hook(grad):
            g, self.backward_res = self.solver(
                lambda y: autograd.grad(
                    f0, z0, y, retain_graph=True)[0] + grad,
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
            H[:, 1: n + 1, 1: n + 1] = (
                torch.bmm(G, G.transpose(1, 2))
                + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[None]
            )
            alpha = torch.solve(y[:, : n + 1], H[:, : n + 1, : n + 1])[0][
                :, 1: n + 1, 0
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

