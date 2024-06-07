import torch
import numpy as np 
import ipdb

def _safe_norm(v):
    if not torch.isfinite(v).all():
        return np.inf
    return torch.norm(v)


def init_fn(x, z_size):
    # mu, z, x = x[:, :z_size], x[:, z_size:2*z_size], x[:, 2*z_size:]
    mu, z, x = x[:, :z_size], x[:, z_size:-z_size], x[:, -z_size:]
    return torch.cat([-mu, - z,  -x], dim=1), torch.cat([mu, z, x], dim=1)

def rmatvec(part_Us, part_VTs, x, init=1):
    # Compute x^T(-I + UV^T)
    # x: (N, 2d, L')
    # part_Us: (N, 2d, L', threshold)
    # part_VTs: (N, threshold, 2d, L')
    if init >1:
        init_x, x = init_fn(x, init)
    else:
        init_x = -init*x
    if part_Us.nelement() == 0:
        return init_x
    xTU = torch.einsum('bij, bijd -> bd', x, part_Us)   # (N, threshold)
    return init_x + torch.einsum('bd, bdij -> bij', xTU, part_VTs)    # (N, 2d, L'), but should really be (N, 1, (2d*L'))


def matvec(part_Us, part_VTs, x, init=1):
    # Compute (-I + UV^T)x
    # x: (N, 2d, L')
    # part_Us: (N, 2d, L', threshold)
    # part_VTs: (N, threshold, 2d, L')
    if init>1:
        init_x, x = init_fn(x, init)
    else:
        init_x = -init*x
    if part_Us.nelement() == 0:
        return init_x
    VTx = torch.einsum('bdij, bij -> bd', part_VTs, x)  # (N, threshold)
    return init_x + torch.einsum('bijd, bd -> bij', part_Us, VTx)     # (N, 2d, L'), but should really be (N, (2d*L'), 1)


def broyden(g, x0, max_steps=20, tol=1e-5, init=1, ls=False, name="unknown", idx=False, x_size=None, printi=True):
    bsz, total_hsize, n_elem = x0.size()
    dev = x0.device
    
    x_est = x0           # (bsz, 2d, L')
    if idx:
        gx = g(x_est, 0)        # (bsz, 2d, L')
    else:
        gx = g(x_est)
    nstep = 0
    tnstep = 0
    LBFGS_thres = min(max_steps, 20)
    
    # For fast calculation of inv_jacobian (approximately)
    Us = torch.zeros(bsz, total_hsize, n_elem, LBFGS_thres).to(dev)
    VTs = torch.zeros(bsz, LBFGS_thres, total_hsize, n_elem).to(dev)
    update = -matvec(Us[:,:,:,:nstep], VTs[:,:nstep], gx, init)# -gx
    new_objective = init_objective = torch.norm(gx).item()
    prot_break = False
    trace = [init_objective]
    new_trace = [-1]
    
    # To be used in protective breaks
    protect_thres = 1e6 * n_elem
    lowest = new_objective
    lowest_xest, lowest_gx, lowest_step = x_est, gx, nstep
    
    while new_objective >= tol and nstep < max_steps:
        if idx:
            g1 = lambda x: g(x, nstep)
        else:
            g1 = g
        x_est, gx, delta_x, delta_gx, ite = line_search(update, x_est, gx, g1, nstep=nstep, on=ls)
        nstep += 1
        tnstep += (ite+1)
        new_objective = torch.norm(gx).item()
        trace.append(new_objective)
        try:
            new2_objective = torch.norm(delta_x).item() / (torch.norm(x_est - delta_x).item())   # Relative residual
        except:
            new2_objective = torch.norm(delta_x).item() / (torch.norm(x_est - delta_x).item() + 1e-9)
        new_trace.append(new2_objective)
        if new_objective < lowest:
            lowest_xest, lowest_gx = x_est.clone().detach(), gx.clone().detach()
            lowest = new_objective
            lowest_step = nstep
        if new_objective < tol:
            break
        if new_objective < 3*tol and nstep > 30 and np.max(trace[-30:]) / np.min(trace[-30:]) < 1.3:
            # if there's hardly been any progress in the last 30 steps
            break
        if new_objective > init_objective * protect_thres:
            prot_break = True
            break

        part_Us, part_VTs = Us[:,:,:,:(nstep-1)], VTs[:,:(nstep-1)]
        # uncomment depending on good broyden vs bad broyden : both usually work.
        vT = delta_gx                                     # good broyden
        # vT = rmatvec(part_Us, part_VTs, delta_x, init)  # bad broyden 
        u = (delta_x - matvec(part_Us, part_VTs, delta_gx, init)) / torch.einsum('bij, bij -> b', vT, delta_gx)[:,None,None]
        vT[vT != vT] = 0
        u[u != u] = 0
        VTs[:,(nstep-1) % LBFGS_thres] = vT
        Us[:,:,:,(nstep-1) % LBFGS_thres] = u
        update = -matvec(Us[:,:,:,:nstep], VTs[:,:nstep], gx, init)

    Us, VTs = None, None
    return {"result": lowest_xest,
            "nstep": nstep,
            "tnstep": tnstep,
            "lowest_step": lowest_step,
            "diff": torch.norm(lowest_gx).item(),
            "diff_detail": torch.norm(lowest_gx, dim=1),
            "prot_break": prot_break,
            "trace": trace,
            "new_trace": new_trace,
            "eps": tol,
            "threshold": max_steps,
            "gx": lowest_gx}

def anderson_jiio(f, x0, x_size, m=5, lam=1e-4, threshold=50, eps=1e-5, stop_mode='rel', beta=0.8, acc_type='good', **kwargs):
    """ Anderson acceleration for fixed point iteration. """
    bsz, d, L = x0.shape
    alternative_mode = 'rel' if stop_mode == 'abs' else 'abs'
    X = torch.zeros(bsz, m, d*L, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d*L, dtype=x0.dtype, device=x0.device)
    fi, cost = f(x0, 0)
    fi = fi.reshape(bsz, -1)
    X[:,0], F[:,0] = x0.reshape(bsz, -1), fi
    fi, cost = f(F[:,0].reshape_as(x0), 1)
    fi = fi.reshape(bsz, -1)
    X[:,1], F[:,1] = F[:,0], fi
    
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1

    trace_dict = {'abs': [],
                  'rel': []}

    lowest_dict = {'abs': 1e12*torch.ones_like(F[:,0,0]),
                   'rel': 1e12*torch.ones_like(F[:,0,0])}
    lowest_step_dict = {'abs': np.ones(bsz),
                        'rel': np.ones(bsz)}

    lowest_xest, lowest_gx =  X[:,1].view_as(x0).clone().detach(), X[:,1].view_as(x0).clone().detach()*0

    lowest_cost = cost
    time1_ = []
    time2_ = []
    time3_ = []

    for k in range(2, threshold):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        if acc_type == 'good':
            if k>2:
                H[:,(k-1)%m+1,1:n+1] = torch.bmm(X[:,(k-1)%m].unsqueeze(1),G.transpose(1,2)).squeeze(1)
                H[:,1:n+1,(k-1)%m+1] = torch.bmm(X[:,:n],G[:,(k-1)%m].unsqueeze(1).transpose(1,2)).squeeze(-1)
            else:
                H[:,1:n+1,1:n+1] = torch.bmm(X[:,:n],G.transpose(1,2))
        else:
            if k>2:
                H[:,(k-1)%m+1,1:n+1] = torch.bmm(G[:,(k-1)%m].unsqueeze(1),G.transpose(1,2)).squeeze(1)
                H[:,1:n+1,(k-1)%m+1] = torch.bmm(G[:,:n],G[:,(k-1)%m].unsqueeze(1).transpose(1,2)).squeeze(-1)
            else:
                H[:,1:n+1,1:n+1] = torch.bmm(G[:,:n],G.transpose(1,2))

        # Could just do alpha = ...
        # But useful when working with some weird scenarios. Helps with ill-conditioned H
        while True:
            try:
                alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])   # (bsz x n)
                break
            except:
                lam = lam*10
                H[:,1:n+1,1:n+1] += lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]

        alpha = alpha[0][:, 1:n+1, 0]
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]

        fi, cost = f(X[:,k%m].reshape_as(x0), k)
        F[:,k%m] = fi.reshape(bsz, -1)
        gx = (F[:,k%m] - X[:,k%m]).view_as(x0)
        diff_x = gx[:, :x_size].norm().item()
        abs_diff = gx.norm(dim=1).squeeze(dim=-1)
        rel_diff = abs_diff / (1e-5 + F[:,k%m].norm(dim=1))
        diff_dict = {'abs': abs_diff,
                     'rel': rel_diff}

        # forward pass criterion : always accept updates for the first 10 steps
        #                        : then make a tradeoff between lower cost and lower kkt residual
        dict_mask = torch.logical_or(diff_dict[stop_mode] < lowest_dict[stop_mode], torch.tensor(k<10))
        mask = torch.logical_or(torch.logical_or(diff_dict[stop_mode] < lowest_dict[stop_mode], torch.tensor(k<10)),torch.logical_and(cost < lowest_cost, diff_dict[stop_mode] < 1.3*lowest_dict[stop_mode]))
        lowest_xest[mask] = X[mask,k%m].clone().detach().unsqueeze(-1)
        lowest_gx[mask] = gx[mask].clone().detach()
        lowest_dict[stop_mode][dict_mask] = diff_dict[stop_mode][dict_mask]
        lowest_step_dict[stop_mode][mask.cpu().numpy()] = k
        lowest_cost[mask] = cost[mask].clone().detach()

    out = {"result": lowest_xest,
           "gx" : lowest_gx,
           "lowest": lowest_dict[stop_mode],
           "nstep": lowest_step_dict[stop_mode],
           "prot_break": False,
           "abs_trace": trace_dict['abs'],
           "rel_trace": trace_dict['rel'],
           "eps": eps,
           "threshold": threshold}
    X = F = None
    return out

def anderson(f, x0, m=5, lam=1e-4, max_steps=20, tol=1e-2, stop_mode='rel', acc_type='good', beta=0.8, solver_name='forward', **kwargs):
    """ 
    Anderson acceleration for fixed point iteration.
    Experimenting with other stopping criterion 
    """
    bsz, d, L = x0.shape
    alternative_mode = 'rel' if stop_mode == 'abs' else 'abs'

    X = torch.zeros(bsz, m, d*L, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d*L, dtype=x0.dtype, device=x0.device)
    fi = f(x0)
    fi = fi.reshape(bsz, -1)
    X[:,0], F[:,0] = x0.reshape(bsz, -1), fi
    fi = f(F[:,0].reshape_as(x0))
    fi = fi.reshape(bsz, -1)
    X[:,1], F[:,1] = F[:,0], fi    
    
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1

    trace_dict = {'abs': [],
                  'rel': []}

    lowest_dict = {'abs': 1e12*torch.ones_like(F[:,0,0]),
                   'rel': 1e12*torch.ones_like(F[:,0,0])}
    lowest_step_dict = {'abs': np.ones(bsz),
                        'rel': np.ones(bsz)}

    lowest_xest, lowest_gx =  X[:,1].view_as(x0).clone().detach(), X[:,1].view_as(x0).clone().detach()*0

    # lowest_cost = cost
    time1_ = []
    time2_ = []
    time3_ = []

    for k in range(2, max_steps):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        if acc_type == 'good':
            H[:,1:n+1,1:n+1] = torch.bmm(X[:,:n],G.transpose(1,2))
        else:
            H[:,1:n+1,1:n+1] = torch.bmm(G[:,:n],G.transpose(1,2))
        while True:
            try:
                alpha = torch.linalg.solve(H[:,:n+1,:n+1], y[:,:n+1])   # (bsz x n)
                break
            except:
                # ipdb.set_trace()
                lam = lam*10
                H[:,1:n+1,1:n+1] += lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
        # ipdb.set_trace()
        alpha = alpha[:, 1:n+1, 0]
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]

        fi = f(X[:,k%m].reshape_as(x0))
        F[:,k%m] = fi.reshape(bsz, -1)
        gx = (F[:,k%m] - X[:,k%m])#.view_as(x0)
        abs_diff = gx.norm(dim=1)#.squeeze(dim=-1)
        rel_diff = abs_diff / (1e-5 + F[:,k%m].norm(dim=1))
        diff_dict = {'abs': abs_diff,
                     'rel': rel_diff}
        # mask = torch.logical_or(diff_dict[stop_mode] < lowest_dict[stop_mode], torch.tensor(k<8))
        mask = diff_dict[stop_mode] < lowest_dict[stop_mode]
        lowest_xest[mask] = X[:,k%m].view_as(x0)[mask].clone().detach()
        lowest_gx[mask] = gx.view_as(x0)[mask].clone().detach()
        lowest_dict[stop_mode][mask] = diff_dict[stop_mode][mask]
        lowest_step_dict[stop_mode][mask.cpu().numpy()] = k
        # lowest_cost[mask] = cost[mask].clone().detach()
        exit_crit = diff_dict[stop_mode] < tol
        # print(f"{solver_name} deq iter : {k}, mean : {diff_dict[stop_mode].mean().item()}, min : {diff_dict[stop_mode].min().item()}, max : {diff_dict[stop_mode].max().item()}")
        if exit_crit.all():
            break

    out = {"result": lowest_xest,
           "gx" : lowest_gx,
           "lowest": lowest_dict[stop_mode],
           "nstep": lowest_step_dict[stop_mode],
           "prot_break": False,
           "abs_trace": trace_dict['abs'],
           "rel_trace": trace_dict['rel'],
           "tol": tol,
           "max_steps": max_steps}
    X = F = None
    return lowest_xest, lowest_gx.mean(), out["nstep"].mean(), lowest_dict[stop_mode].mean()