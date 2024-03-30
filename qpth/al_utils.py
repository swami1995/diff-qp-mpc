import torch
from collections import namedtuple
import ipdb

QuadCost = namedtuple('QuadCost', 'C c')
LinDx = namedtuple('LinDx', 'F f')

# https://stackoverflow.com/questions/11351032
QuadCost.__new__.__defaults__ = (None,) * len(QuadCost._fields)
LinDx.__new__.__defaults__ = (None,) * len(LinDx._fields)

@torch.jit.script
def merit_function(xu, Q, q, dx, x0, lamda, rho, diag_cost=True, grad=False):
    bsz = xu.size(0)
    cost_total = compute_cost(xu, Q, q)
    res, res_clamp = dyn_res(xu, dx, x0, res_type='both')
    # if grad:
        # return cost_total +  0.5*rho[:,0]*(res_clamp*res_clamp).view(bsz, -1).sum(dim=1)
    # else:
    return cost_total + (lamda*res).view(bsz, -1).sum(dim=1) + 0.5*rho[:,0]*(res_clamp*res_clamp).view(bsz, -1).sum(dim=1)
    # else:
    #     return cost_total.mean() + (lamda*res_clamp).view(bsz, -1).sum(dim=1).mean() + 0.5*rho*(res_clamp*res_clamp).view(bsz, -1).sum(dim=1).mean()

@torch.jit.script
def merit_cost_grad_hessian(xu, Q, q, dx, dx_jac, x0, lamda, rho, x_lower, x_upper, u_lower, u_upper, ineqG, ineqh, diag_cost=True):
    bsz = xu.size(0)
    xi_shape = (1, xu.shape[1], xu.shape[2])
    # res, res_clamp = self.dyn_res(xu, dx, x0, res_type='both')
    # dyn_res_fn = lambda xi, xi0 : dyn_res(xi.view(xi_shape), dx, xi0.view(1, x0.shape[1]), res_type='clamp')[0]
    # constraint jacobian
    # constraint_jac = vmap(jacrev(dyn_res_fn))(xu.view(bsz, -1), x0)#, self.mask)
    res, res_clamp, constraint_jac, constraint_jac_clamp = constraint_res_jac1(xu, x0, dx, x_lower, x_upper, u_lower, u_upper, ineqG, ineqh)
    constraint_hess = torch.bmm(constraint_jac_clamp.permute(0,2,1), constraint_jac_clamp)
    cost_total = compute_cost(xu, Q, q, diag_cost)
    merit = cost_total + (lamda*res).view(bsz, -1).sum(dim=1) + 0.5*rho[:,0]*(res_clamp*res_clamp).view(bsz, -1).sum(dim=1)
    merit_grad = compute_cost_gradient(xu, Q, q, diag_cost) + (lamda[...,None]*constraint_jac).sum(dim=-2) + rho*(res_clamp[...,None]*constraint_jac_clamp).sum(dim=-2)
    if diag_cost:
        Qfull = torch.diag_embed(Q.reshape(bsz, -1))
    merit_hess = Qfull + rho[:,:,None]*constraint_hess
    return merit, merit_grad, merit_hess

@torch.jit.script
def merit_hessian(xu, Q, q, dx, dx_jac, x0, lamda, rho, x_lower, x_upper, u_lower, u_upper, diag_cost=True):
    bsz = xu.size(0)
    xi_shape = (1, xu.shape[1], xu.shape[2])
    # res, res_clamp = self.dyn_res(xu, dx, x0, res_type='both')
    # dyn_res_fn = lambda xi, xi0 : dyn_res(xi.view(xi_shape), dx, xi0.view(1, x0.shape[1]), res_type='clamp')[0]
    # constraint jacobian
    # constraint_jac = vmap(jacrev(dyn_res_fn))(xu.view(bsz, -1), x0)#, self.mask)
    res, res_clamp, constraint_jac, constraint_jac_clamp = constraint_res_jac1(xu, x0, dx, x_lower, x_upper, u_lower, u_upper)
    constraint_hess = torch.bmm(constraint_jac_clamp.permute(0,2,1), constraint_jac_clamp)
    if diag_cost:
        Qfull = torch.diag_embed(Q.reshape(bsz, -1))
    merit_hess = Qfull + rho[:,:,None]*constraint_hess
    return merit_hess

@torch.jit.script
def constraint_res_jac1(xu, x0, dx, x_lower, x_upper, u_lower, u_upper):
    x_size = x0.size(-1)
    bsz, T, xu_size = xu.shape
    u_size = xu_size - x_size
    neq = x_size*T
    n_ineq = 2*u_size*T
    n_constr = neq + n_ineq
    # xu = xu.view(bsz, -1)
    # x0 = x0.view(bsz, -1)
    x, u = x[:,:,:x_size], x[:,:,x_size:]
    x = x.view(1, bsz, T, x_size).repeat(n_constr, 1, 1, 1).view(n_constr*bsz, T, x_size)
    u = u.view(1, bsz, T, u_size).repeat(n_constr, 1, 1, 1).view(n_constr*bsz, T, u_size)
    res_eq = dyn_res_eq(x, u, dx, x0).reshape(bsz, n_constr, neq)
    res_ineq, res_ineq_clamp = dyn_res_ineq(x, u, dx, x0, x_lower, x_upper, u_lower, u_upper)
    res_ineq = res_ineq.reshape(bsz, n_constr, n_ineq)
    res_ineq_clamp = res_ineq_clamp.reshape(bsz, n_constr, n_ineq)
    res = torch.cat((res_eq, res_ineq), dim=-1)
    identity = torch.eye(n_constr, device=x.device)[None].to(x).repeat(bsz, 1, 1)
    res = res*identity
    constraint_jac = torch.autograd.grad([res.sum()], [x, u])
    constraint_jac = torch.cat((constraint_jac[0].reshape(bsz, n_constr, T, x_size), constraint_jac[1].reshape(bsz, n_constr, T, u_size)), dim=-1).view(bsz, n_constr, T*xu_size)
    res = res[:,0]
    res_clamp = torch.cat((res_eq[:,0], res_ineq_clamp[:,0]), dim=1)
    mask = (res_ineq_clamp[:,0] > 0).float()
    constraint_jac_clamp = constraint_jac * mask[...,None] 
    return res, res_clamp, constraint_jac, constraint_jac_clamp

@torch.jit.script
def constraint_res_jac2(xu, x0, dx, x_lower, x_upper, u_lower, u_upper):
    x_size = x0.size(-1)
    bsz, T, xu_size = xu.shape
    u_size = xu_size - x_size
    # neq = x_size*T
    # n_ineq = 2*u_size*T
    # n_constr = neq + n_ineq
    # xu = xu.view(bsz, -1)
    # x0 = x0.view(bsz, -1)
    x, u = x[:,:,:x_size], x[:,:,x_size:]

    res_eq, res_eq_jac = dyn_res_eq_jac()
    res_ineq, res_ineq_clamp, res_ineq_jac, res_ineq_jac_clamp = dyn_res_ineq(x, u, dx, x0, x_lower, x_upper, u_lower, u_upper)
    res = torch.cat((res_eq, res_ineq), dim=-1)
    res_clamp = torch.cat((res_eq, res_ineq_clamp), dim=-1)
    constraint_jac = torch.cat((res_eq_jac, res_ineq_jac), dim=1)
    constraint_jac_clamp = torch.cat((res_eq_jac, res_ineq_jac_clamp), dim=1)
    return res, res_clamp, constraint_jac, constraint_jac_clamp

@torch.jit.script
def dyn_res_eq(x, u, dx, x0, mask=None):
    " split x into state and control and compute dynamics residual using dx"
    bsz = x.size(0)
    # if isinstance(dx, LinDx):
    #     x_next = (dx.F.permute(1,0,2,3)*torch.cat((x, u), dim=2)[:,:-1,None,:]).sum(dim=-1) + dx.f.permute(1,0,2)
    # else:
    x_next = dx(x, u)[:,:-1]
    res = (x[:,1:,:] - x_next)
    res_init = (x[:,0,:] - x0).reshape(bsz, 1, -1)
    # print(res.shape, res_init.shape, x0.shape, x.shape)
    # if self.add_goal_constraint:
    #     res_goal = (x[:,-1,:] - self.x_goal).reshape(bsz, -1)
    #     res = torch.cat((res, res_init, res_goal), dim=1)
    # else:
    res = torch.cat((res, res_init), dim=1)
    res = res.reshape(bsz, -1)
    return res

@torch.jit.script
def dyn_res_eq_jac(x, u, dx, x0):
    bsz, T, x_size = x.shape
    u_size = u.shape[-1]
    x = x.view(bsz, T, 1, x_size).repeat(1, 1, x_size, 1).view(bsz, T*x_size, x_size)
    u = u.view(bsz, T, 1, u_size).repeat(1, 1, x_size, 1).view(bsz, T*x_size, u_size)
    x_next = dx(x, u).view(bsz, T, x_size, x_size)[:,:-1]
    identity = torch.eye(x_size, device=x.device)[None].to(x).repeat(bsz, T-1, 1, 1)
    dyn_out = x_next*identity
    dynamics_jacobian = torch.autograd.grad([dyn_out.sum()], [x, u])
    dynamics_jacobian = torch.cat((dynamics_jacobian[0].reshape(bsz, T-1, x_size, x_size), dynamics_jacobian[1].reshape(bsz, T-1, x_size, u_size)), dim=-1)
    # dynamics_jacobian = torch.vmap(block_diag)(dynamics_jacobian)
    dynamics_jacobian = torch.stack([torch.block_diag(*dynamics_jacobian[i]) for i in range(bsz)])

    x_res = x[:,1:,:] - x_next
    x_res = x_res.reshape(bsz, -1)
    x_res_init = x[:,0,:] - x0
    x_res_init = x_res_init.reshape(bsz, -1)
    res = torch.cat((x_res, x_res_init), dim=1)

    constraint_jacobian = torch.zeros(bsz, T*x_size, T*(x_size+u_size)).to(x)
    id_x = [torch.cat([torch.eye(x_size), torch.zeros((x_size, u_size))], dim=1).to(x)]*T-1
    id_x = torch.block_diag(*id_x)
    id_x = id_x[None].repeat(bsz, 1, 1)
    constraint_jacobian[:,:-x_size,x_size+u_size:] = id_x
    constraint_jacobian[:,:-x_size,:-(x_size+u_size)] = -dynamics_jacobian
    constraint_jacobian[:, -x_size:, :x_size] = torch.eye(x_size).to(x)[None]
    return res, constraint_jacobian


@torch.jit.script
def dyn_res_ineq(x, u, dx, x0, x_lower, x_upper, u_lower, u_upper):
    # Add state constraints if self.x_lower and self.x_upper are not None
    bsz = x.size(0)
    res = None
    # if x_lower is not None:
    #     res = torch.cat((
    #         x - x_upper, 
    #         -x + x_lower
    #         ), dim=2)

    # Add control constraints if self.u_lower and self.u_upper are not None
    # if u_lower is not None:
    res = torch.cat((
        u - u_upper,
        -u + u_lower
        ), dim=2)
        # if res is None:
        #     res = res_u
        # else:
        #     res = torch.cat((res,res_u), dim=2)
    
    # Add other inequality constraints if self.ineqG and self.ineqh are not None
    # if self.ineqG is not None:
    #     res_G = torch.bmm(self.ineqG, x.unsqueeze(-1)) - self.ineqh
    #     if res is None:
    #         res = res_G
    #     else:
    #         res = torch.cat((res,res_G), dim=2)
    res = res.reshape(bsz, -1)
    res_clamp = torch.clamp(res, min=0)
    return res, res_clamp

def dyn_res_ineq_jac(x, u, dx, x0, x_lower, x_upper, u_lower, u_upper):
    bsz, T, x_size = x.shape
    u_size = u.shape[-1]
    res = None
    res = torch.cat((
        u - u_upper,
        -u + u_lower
        ), dim=2)
    res = res.reshape(bsz, -1)
    res_clamp = torch.clamp(res, min=0)
    id_u = torch.eye(u_size, device=x.device).to(x)
    id_u = torch.cat((id_u, -id_u), dim=0)
    id_x = torch.zeros((2*u_size, x_size), device=x.device).to(x)
    id_xu = torch.cat((id_x, id_u), dim=1)
    id_xu = [id_xu]*T
    id_xu = torch.block_diag(*id_xu)
    id_xu = id_xu[None].repeat(bsz, 1, 1)
    id_xu_clamp = id_xu * (res_clamp > 0).float()[...,None]
    return res, res_clamp, id_xu, id_xu_clamp

@torch.jit.script
def dyn_res(self, x, dx, x0, res_type='clamp'):
    x = x.reshape(-1, self.T, self.n_state+self.n_ctrl)
    x, u = x[:,:,:self.n_state], x[:,:,self.n_state:]
    # Equality residuals
    res_eq = self.dyn_res_eq(x, u, dx, x0)#, mask)
    # Inequality residuals
    res_ineq, res_ineq_clamp = self.dyn_res_ineq(x, u, dx, x0)
    if res_type == 'noclamp':
        return torch.cat((res_eq, res_ineq), dim=1)
    elif res_type == 'clamp':
        return torch.cat((res_eq, res_ineq_clamp), dim=1)
    else:
        return torch.cat((res_eq, res_ineq), dim=1), torch.cat((res_eq, res_ineq_clamp), dim=1)

@torch.jit.script
def grad_res(self, x, y, Q, q, dx, x0, lam, rho, y_update=False):
    " compute the gradient of the residual with respect to x"
    # ipdb.set_trace()
    if y_update:
        return lam + rho*self.dyn_res(x, dx, x0)
    cost_grad = self.compute_cost_gradient(x, Q, q)
    x.requires_grad_(True)
    with torch.enable_grad():
        dyn_res = self.dyn_res(x, dx, x0)
        grad_res = torch.autograd.grad(dyn_res, x, grad_outputs=y)[0]
    x_res = grad_res + cost_grad
    # y_res = y/rho - lam/rho - dyn_res
    # dr = torch.cat((x_res, y_res), dim=-1)
    # print("residual norms : ", dr.norm(), cost_grad.norm(), dyn_res.norm())
    # if y_update:
    #     return lam + rho*dyn_res
    return x_res

@torch.jit.script
def compute_cost(self, xu, Q, q, diag_cost=True):
    # ipdb.set_trace()
    C = Q.transpose(0,1)
    c = q.transpose(0,1)
    if diag_cost:
        return (0.5*( xu * C * xu ).sum(-1) + ( c * xu ).sum(-1)).sum(dim=-1)
    return 0.5*((xu.unsqueeze(-1)*C).sum(dim=-2)*xu).sum(dim=-1).sum(dim=-1) + (xu*c).sum(dim=-1).sum(dim=-1)

@torch.jit.script
def compute_cost_gradient(self, xu, Q, q, diag_cost=True):
    # ipdb.set_trace()
    C = Q.transpose(0,1)
    c = q.transpose(0,1)
    if diag_cost:
        return C*xu + c
    return torch.cat((C*xu, c), dim=-1)

class NewtonAL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, meritfn, dyn_fn, cost_fn, merit_hess, xi, x0, z, lam, rho, Q, q, Hess, threshold, eps, ls):
        bsz, T, n_elem = xi.size() # (bsz, T, xd+ud)
        dev = xi.device

        meritfnQ = lambda x, grad=False : meritfn(x, Q, q, lam, grad=grad)
        gQ = lambda x, z, grad=False : g(x, z, Q, q, grad)
        cost_fnQ = lambda x : cost_fn(x, Q, q)
        meritfn_mean = lambda x, Qi, qi, yi, x0i, rhoi: meritfn(x.view((1,T,n_elem)), Qi[None].transpose(0,1).view((1,T,n_elem)), qi[None].transpose(0,1).view((1,T,n_elem)), yi[None], x0i[None], rhoi[None], grad=True).mean()

        x_est = xi           # (bsz, 2d, L')
        gx = gQ(x_est, z)  # (bsz, 2d, L')
        cost = cost_fnQ(x_est) 
        dyn_res = dyn_fn(x_est)
        tnstep = 0

        # Solve for newton steps on the augmented lagrangian
        nstep = 0
        prot_break = False
        lowest_gx = gx
        lowest_step = 0
        old_dyn_res = torch.norm(dyn_res).item()
        # print(nstep, torch.norm(dyn_res).item(), torch.norm(cost).item(), torch.norm(gx).item())
        init_update_norm = gx.norm().item()
        update_norm = init_update_norm
        Q_tr = Q.transpose(0,1)
        q_tr = q.transpose(0,1)
        stepsz = 1
        # reg = torch.ones(bsz, device=dev, dtype=xi.dtype)*1e-8
        # status = torch.ones(bsz, device=dev, dtype=xi.dtype)
        rho_new = rho
        # Hess = vmap(hessian(meritfn_mean))(x_est.reshape(bsz, -1), Q_tr, q_tr, lam, x0, rho_new).reshape(bsz, T*3,T*3)
        # Hess = merit_hess(x_est)
        # U, info = torch.linalg.cholesky_ex(Hess)
        # U = None
        cholesky_fail = torch.tensor(False)
        # if torch.any(info):
        #     cholesky_fail = torch.tensor(True)
            # Hesses = Hess[info > 0]
            # for H in Hesses:
            #     eigs = torch.linalg.eigvals(H).real
            #     neg_eigs = eigs[eigs < 0]
            #     if neg_eigs.nelement() > 0:
            #         print(neg_eigs)
            # print("Cholesky failed")
        # ipdb.set_trace()
        while torch.norm(gx).item() > threshold and nstep < 4 and stepsz > 1e-8:# and update_norm > 1e-3*init_update_norm:
            nstep += 1
            
            # Compute the hessian and gradient of the augmented lagrangian
            # ipdb.set_trace()
            with torch.enable_grad():
                x_est.requires_grad_(True)
                merit = meritfnQ(x_est)#, grad=True)
                merit_mean = merit.sum()
                grad = torch.autograd.grad(merit_mean, x_est)[0]
                Hess = merit_hess(x_est)
                # Hess = hessian(meritfn_mean, (x_est.reshape(bsz, -1)), vectorize=True)
                # Hess = vmap(hessian(meritfn_mean))(x_est.reshape(bsz, -1), Q_tr, q_tr, lam, x0, rho_new)
            # Hess = Hess + torch.eye(total_hsize, device=dev).unsqueeze(0).expand(bsz, total_hsize, total_hsize) * eps
            
            # Solve for the newton step
            stepsz = 0
            reg = 0#reg*10*(1-status) + status*1e-5
            # Hess = Hess.reshape(bsz, T*3,T*3)
            # eye = torch.eye(T*3, device=dev).to(Hess).unsqueeze(0).expand(bsz, T*3, T*3)
            # while stepsz < 1e-5:
            if not cholesky_fail:
                U, info = torch.linalg.cholesky_ex(Hess)
                update = -torch.cholesky_solve(grad.reshape(bsz, -1, 1), U).reshape(bsz,T,n_elem)
                if update.isnan().sum() > 0 or update.isinf().sum() > 0:
                    # ipdb.set_trace()
                    update = -torch.linalg.solve(Hess, grad.reshape(bsz, -1)).reshape(bsz,T,n_elem)
                    cholesky_fail = torch.tensor(True)
            else:
                update = -torch.linalg.solve(Hess, grad.reshape(bsz, -1)).reshape(bsz,T,n_elem)
            if ls:
                x_est, new2_objective, stepsz, status = line_search_newton(update, x_est, meritfnQ, merit)
            else:
                x_est = x_est + update
                new2_objective = meritfnQ(x_est).mean().item()
            # reg *= 10
            # x_est = x_est + stepsz * update
            # gx = gQ(x_est, z)
            # cost = cost_fnQ(x_est)
            if x_est.isnan().sum() > 0 or x_est.isinf().sum() > 0:
                ipdb.set_trace()
            dyn_res = dyn_fn(x_est)
            # update_norm = update.norm().item()
            new_dyn_res = torch.norm(dyn_res).item()
            # print(nstep, torch.norm(dyn_res).item(), torch.norm(cost).item(), new2_objective, torch.norm(gx).item(), torch.norm(update).item(), stepsz)

            ## exit creteria
            if abs(old_dyn_res- new_dyn_res)/new_dyn_res < 1e-3 or new_dyn_res < 1e-3:
                break
                
            old_dyn_res = new_dyn_res
            # rho_new = rho*status[:,None] + rho_new/2*(1-status[:,None])#min(, rho_init*100)
        # print(nstep)
        try:
            ctx.save_for_backward(Hess, U, x_est, cholesky_fail)
        except:
            ipdb.set_trace()
        Us, VTs = None, None
        return x_est, gx, status
    
    @staticmethod
    def backward(ctx, x_grad, gx_grad, status_grad):
        # implicit gradients w.r.t Q and q
        H, U, x, cholesky_fail = ctx.saved_tensors
        bsz = x_grad.size(0)

        # solve Hx + g = 0, H = d^2f/dx^2, g is x_grad
        if cholesky_fail:
            inp_grad = -torch.linalg.solve(H, x_grad.view(bsz, -1)).reshape(x_grad.shape)
        else:
            inp_grad = -torch.cholesky_solve(x_grad.view(bsz, -1, 1), U).reshape(x_grad.shape)

        # Compute the gradient w.r.t. the Q and q 
        Q_grad = inp_grad*x # if Q is diag
        # Q_grad = torch.bmm(inp_grad, x.transpose(1,2)) # if Q is not diag
        q_grad = inp_grad

        return None, None, None, None, None, None, None, None, None, None, Q_grad.transpose(0,1), q_grad.transpose(0,1), None, None, None, None

def line_search_newton(update, x_est, meritfnQ, merit):
    stepsz = torch.ones(x_est.shape[0], device=x_est.device)*2
    mask = torch.ones(x_est.shape[0], device=x_est.device)
    # while mask.sum() > 0 and stepsz.min() > 1e-6:
    #     stepsz = 0.5*stepsz*mask.float() + stepsz*(1-mask.float())
    #     x_next = x_est + stepsz[:,None,None] * update
    #     new2_objective = meritfnQ(x_next)
    #     mask = new2_objective > merit
    stepszs = 2**(-torch.arange(20, device=x_est.device).float().unsqueeze(1).expand(20, x_est.shape[0]))
    x_next = (x_est[None] + stepszs[:,:,None,None] * update[None])
    new2_objective = torch.vmap(meritfnQ)(x_next)
    # ipdb.set_trace()
    new2_objective_min = torch.min(new2_objective, dim=0)
    batch_idxs = torch.arange(x_est.shape[0], device=x_est.device)
    stepsz = stepszs[new2_objective_min.indices, batch_idxs]
    x_next = x_next[new2_objective_min.indices, batch_idxs]
    new2_objective = new2_objective_min.values
    status = (new2_objective < merit).float()
    # if not status.all():
    #     print("Warning: Line search failed")
    x_est = status[:,None,None] * x_next + (1-status)[:,None,None] * x_est
    return x_est, new2_objective.mean().item(), stepsz.mean().item(), status
      