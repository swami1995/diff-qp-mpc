import torch
import numpy as np
import ipdb
import time

from torch.autograd import Variable
from torch.func import hessian, vmap

def print_header(msg):
    print('===>', msg)


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().numpy()


def get_sizes(G, A=None):
    if G.dim() == 2:
        nineq, nz = G.size()
        nBatch = 1
    elif G.dim() == 3:
        nBatch, nineq, nz = G.size()
    if A is not None:
        neq = A.size(1) if A.nelement() > 0 else 0
    else:
        neq = None
    # nBatch = batchedTensor.size(0) if batchedTensor is not None else None
    return nineq, nz, neq, nBatch


def expandParam(X, nBatch, nDim):
    if X.ndimension() in (0, nDim) or X.nelement() == 0:
        return X, False
    elif X.ndimension() == nDim - 1:
        return X.unsqueeze(0).expand(*([nBatch] + list(X.size()))), True
    else:
        raise RuntimeError("Unexpected number of dimensions.")


def extract_nBatch(Q, p, G, h, A, b):
    dims = [3, 2, 3, 2, 3, 2]
    params = [Q, p, G, h, A, b]
    for param, dim in zip(params, dims):
        if param.ndimension() == dim:
            return param.size(0)
    return 1


import operator

def jacobian(f, x, eps):
    if x.ndimension() == 2:
        assert x.size(0) == 1
        x = x.squeeze()

    e = Variable(torch.eye(len(x)).type_as(get_data_maybe(x)))
    J = []
    for i in range(len(x)):
        J.append((f(x + eps*e[i]) - f(x - eps*e[i]))/(2.*eps))
    J = torch.stack(J).transpose(0,1)
    return J


def expandParam(X, n_batch, nDim):
    if X.ndimension() in (0, nDim):
        return X, False
    elif X.ndimension() == nDim - 1:
        return X.unsqueeze(0).expand(*([n_batch] + list(X.size()))), True
    else:
        raise RuntimeError("Unexpected number of dimensions.")


def bdiag(d):
    assert d.ndimension() == 2
    nBatch, sz = d.size()
    dtype = d.type() if not isinstance(d, Variable) else d.data.type()
    D = torch.zeros(nBatch, sz, sz).type(dtype)
    I = torch.eye(sz).repeat(nBatch, 1, 1).type(dtype).byte()
    D[I] = d.view(-1)
    return D


def bger(x, y):
    return x.unsqueeze(2).bmm(y.unsqueeze(1))


def bmv(X, y):
    return X.bmm(y.unsqueeze(2)).squeeze(2)


def bquad(x, Q):
    return x.unsqueeze(1).bmm(Q).bmm(x.unsqueeze(2)).squeeze(1).squeeze(1)


def bdot(x, y):
    return torch.bmm(x.unsqueeze(1), y.unsqueeze(2)).squeeze(1).squeeze(1)


def eclamp(x, lower, upper):
    # In-place!!
    if type(lower) == type(x):
        assert x.size() == lower.size()

    if type(upper) == type(x):
        assert x.size() == upper.size()

    I = x < lower
    x[I] = lower[I] if not isinstance(lower, float) else lower

    I = x > upper
    x[I] = upper[I] if not isinstance(upper, float) else upper

    return x


def get_data_maybe(x):
    return x if not isinstance(x, Variable) else x.data


_seen_tables = []
def table_log(tag, d):
    # TODO: There's probably a better way to handle formatting here,
    # or a better way altogether to replace this quick hack.
    global _seen_tables

    def print_row(r):
        print('| ' + ' | '.join(r) + ' |')

    if tag not in _seen_tables:
        print_row(map(operator.itemgetter(0), d))
        _seen_tables.append(tag)

    s = []
    for di in d:
        assert len(di) in [2,3]
        if len(di) == 3:
            e, fmt = di[1:]
            s.append(fmt.format(e))
        else:
            e = di[1]
            s.append(str(e))
    print_row(s)


def get_traj(T, u, x_init, dynamics):
    from .mpc import QuadCost, LinDx # TODO: This is messy.

    if isinstance(dynamics, LinDx):
        F = get_data_maybe(dynamics.F)
        f = get_data_maybe(dynamics.f)
        if f is not None:
            assert f.shape == F.shape[:3]

    x = [get_data_maybe(x_init)]
    for t in range(T):
        xt = x[t]
        ut = get_data_maybe(u[t])
        if t < T-1:
            # new_x = f(Variable(xt), Variable(ut)).data
            if isinstance(dynamics, LinDx):
                xut = torch.cat((xt, ut), 1)
                new_x = bmv(F[t], xut)
                if f is not None:
                    new_x += f[t]
            else:
                new_x = dynamics(Variable(xt), Variable(ut)).data
            x.append(new_x)
    x = torch.stack(x, dim=0)
    return x


def get_cost(T, u, cost, dynamics=None, x_init=None, x=None):
    from .mpc import QuadCost, LinDx # TODO: This is messy.

    assert x_init is not None or x is not None

    if isinstance(cost, QuadCost):
        C = get_data_maybe(cost.C)
        c = get_data_maybe(cost.c)

    if x is None:
        x = get_traj(T, u, x_init, dynamics)

    objs = []
    for t in range(T):
        xt = x[t]
        ut = u[t]
        xut = torch.cat((xt, ut), 1)
        if isinstance(cost, QuadCost):
            obj = 0.5*bquad(xut, C[t]) + bdot(xut, c[t])
        else:
            obj = cost(xut)
        objs.append(obj)
    objs = torch.stack(objs, dim=0)
    total_obj = torch.sum(objs, dim=0)
    return total_obj


def detach_maybe(x):
    if x is None:
        return None
    return x if not x.requires_grad else x.detach()


def data_maybe(x):
    if x is None:
        return None
    return x.data

def init_fn(dr, ysize, rho, Hinv):
    # dx, dy = dr[..., :-ysize], dr[..., -ysize:]
    # return torch.cat([dx*Hinv, -dy*rho], dim=-1), dr
    return dr*Hinv, dr

def matvec(part_Us, part_VTs, x, ysize=1, rho=1, Hinv=1):
    # Compute (-I + UV^T)x
    # x: (N, 2d, L')
    # part_Us: (N, 2d, L', threshold)
    # part_VTs: (N, threshold, 2d, L')
    init_x, x = init_fn(x, ysize, rho, Hinv)
    if part_Us.nelement() == 0:
        return init_x
    VTx = torch.einsum('bdij, bij -> bd', part_VTs, x)  # (N, threshold)
    return init_x + torch.einsum('bijd, bd -> bij', part_Us, VTx)     # (N, 2d, L'), but should really be (N, (2d*L'), 1)


def broyden_AL(g, meritfn, dyn_fn, cost_fn, x0, y, threshold, eps, rho=1, Hinv=1, ysize=1, ls=False, name="unknown", idx=False, x_size=None, printi=True):
    bsz, total_hsize, n_elem = x0.size() # (bsz, T, xd+ud)
    dev = x0.device

    x_est = x0           # (bsz, 2d, L')
    if idx:
        gx = g(x_est, y)        # (bsz, 2d, L')
    else:
        gx = g(x_est)
    nstep = 0
    tnstep = 0
    LBFGS_thres = min(threshold, 20)

    # For fast calculation of inv_jacobian (approximately)
    Us = torch.zeros(bsz, total_hsize, n_elem, LBFGS_thres).to(x0)
    VTs = torch.zeros(bsz, LBFGS_thres, total_hsize, n_elem).to(x0)
    update = -matvec(Us[:,:,:,:nstep], VTs[:,:nstep], gx, ysize, rho, Hinv)# -gx
    new_objective = init_objective = torch.norm(gx).item()
    prot_break = False
    trace = [init_objective]
    new_trace = [-1]

    # To be used in protective breaks
    protect_thres = 1e6 * n_elem
    lowest = new_objective
    lowest_xest, lowest_gx, lowest_step = x_est, gx, nstep
    dyn_res = dyn_fn(x_est) if dyn_fn is not None else 0
    cost = cost_fn(x_est) if cost_fn is not None else 0
    print("nstep, dyn residual, cost , rel residual, merit value, torch.norm(delta_x), torch.norm(gx), torch.norm(update), s")
    print(0, torch.norm(dyn_res).item(), torch.norm(cost).item(), torch.norm(gx).item())
    while new_objective >= eps and nstep < threshold:
        if idx:
            g1 = lambda x: g(x, y)
        else:
            g1 = g
        x_est, gx, delta_x, delta_gx, ite, s, merit = line_search(update, x_est, gx, g1, meritfn, nstep=nstep, on=ls)
        yopt = g(x_est, y, y_update=True)
        y = yopt#*(1-s) + yopt*s
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
        if new_objective < eps:
            print("converged 1")
            break
        if new_objective < 3*eps and nstep > 30 and np.max(trace[-30:]) / np.min(trace[-30:]) < 1.3:
            # if there's hardly been any progress in the last 30 steps
            print("converged 2")
            break
        if new_objective > init_objective * protect_thres:
            prot_break = True
            print("converged 3")
            break

        part_Us, part_VTs = Us[:,:,:,:(nstep-1)], VTs[:,:(nstep-1)]
        # uncomment depending on good broyden vs bad broyden : both usually work.
        vT = delta_gx                                     # good broyden
        # vT = rmatvec(part_Us, part_VTs, delta_x, init)  # bad broyden 
        u = (delta_x - matvec(part_Us, part_VTs, delta_gx, ysize, rho, Hinv)) / torch.einsum('bij, bij -> b', vT, delta_gx)[:,None,None]
        vT[vT != vT] = 0
        u[u != u] = 0
        VTs[:,(nstep-1) % LBFGS_thres] = vT
        Us[:,:,:,(nstep-1) % LBFGS_thres] = u
        update = -matvec(Us[:,:,:,:nstep], VTs[:,:nstep], gx, ysize, rho, Hinv)
        dyn_res = dyn_fn(x_est) if dyn_fn is not None else 0
        cost = cost_fn(x_est) if cost_fn is not None else 0
        print(nstep, torch.norm(dyn_res).item(), torch.norm(cost).item(), new2_objective, merit.mean().item(), torch.norm(delta_x).item(), torch.norm(x_est).item(), torch.norm(gx).item(), torch.norm(update).item(), s)
        # ipdb.set_trace()
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
            "eps": eps,
            "threshold": threshold,
            "gx": lowest_gx}

def GD_AL(g, r0, threshold, eps, rho=1, Hinv=1, ysize=1, ls=False, name="unknown", dyn_fn=None, cost_fn=None):
    " Alternating gradient descent on the augmented Lagrangian for x and y"
    bsz, total_hsize, n_elem = r0.size() # (bsz, T, xd+ud)
    dev = r0.device
    r_est = r0
    betax = 0.1
    betay = 0.4
    x_est = r_est[..., :-ysize]
    y_est = r_est[..., -ysize:]
    dyn_res = dyn_fn(x_est) if dyn_fn is not None else 0
    cost = cost_fn(x_est) if cost_fn is not None else 0
    print(torch.cat([y_est[0], dyn_res[0]], dim=-1))

    for i in range(20):           # (bsz, 2d, L')
        gr = g(r_est)
        x = r_est[..., :-ysize]
        y = r_est[..., -ysize:]
        x_est = x - betax * gr[..., :-ysize]
        y_est = y - betay * gr[..., -ysize:]
        r_est = torch.cat([x_est, y_est], dim=-1)
        delta_x = x_est - x
        delta_y = y_est - y
        dyn_res = dyn_fn(x_est) if dyn_fn is not None else 0
        cost = cost_fn(x_est) if cost_fn is not None else 0
        print(i,  torch.norm(delta_x).item(), torch.norm(delta_y).item(), torch.norm(dyn_res).item(), torch.norm(cost).item(), torch.norm(y_est).item(), torch.norm(gr).item())
        print(torch.cat([y_est[0], dyn_res[0]], dim=-1))
        if torch.norm(gr).item() < eps:
            break
    ipdb.set_trace()
    return {"result": r_est,
            "eps": eps,
            "threshold": threshold,
            "gx": gr}

class iterationData:
    """docstring for iterationData"""
    def __init__(self, alpha, s, y, ys):
        self.alpha = alpha
        self.s = s
        self.y = y
        self.ys = ys

def LBFGS_AL(g, meritfn, dyn_fn, cost_fn, x0, z, threshold, eps, rho=1, Hinv=1, ysize=1, ls=False, name="unknown", idx=False, x_size=None, printi=True):
    bsz, total_hsize, n_elem = x0.size() # (bsz, T, xd+ud)
    dev = x0.device

    x_est = x0           # (bsz, 2d, L')
    gx = g(x_est, z)  # (bsz, 2d, L')
    cost = cost_fn(x_est) 
    dyn_res = dyn_fn(x_est)
    tnstep = 0
    lbfgs_mem = LBFGS_thres = min(threshold, 20)

    # For fast calculation of inv_jacobian (approximately)
    lm = []
    for i in range(0, threshold):
        s = torch.zeros(bsz, total_hsize, dtype=x0.dtype).to(x0.device)
        y = torch.zeros(bsz, total_hsize, dtype=x0.dtype).to(x0.device)
        lm.append(iterationData(s[:,0], s, y, s[:,0]))

    Hinv = Hinv#.clamp(-1, 1)
    update = -Hinv * gx # Need to adaptively regularize Hinv
    new_objective = init_objective = torch.norm(gx).item()
    prot_break = False
    trace = [init_objective]
    new_trace = [-1]

    # To be used in protective breaks
    nstep = 0
    protect_thres = 1e6 * n_elem
    lowest = new_objective
    lowest_xest, lowest_gx, lowest_step = x_est, gx, nstep
    num_fails = 0
    num_fails_total = 0
    print("nstep, dyn residual, cost , rel residual, merit value, torch.norm(delta_x), torch.norm(gx), torch.norm(update), s")
    print(0, torch.norm(dyn_res).item(), torch.norm(cost).item(), torch.norm(gx).item())
    while nstep < threshold:# and new_objective >= eps:
        if idx:
            g1 = lambda x: g(x, z)
        else:
            g1 = g
        gx_old = gx
        x_est, gx, delta_x, delta_gx, ite, stepsz, merit = line_search(update, x_est, gx, g1, meritfn, nstep=nstep, on=ls)
        Bs = -stepsz*gx_old
        zopt = g(x_est, z, y_update=True)
        sz = 1.0#max(stepsz, 0.5)
        z = z*(1-sz) + zopt*sz
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
        # if new_objective < eps:
        #     print("converged 1")
        #     break
        # if new_objective < 3*eps and nstep > 30 and np.max(trace[-30:]) / np.min(trace[-30:]) < 1.3:
        #     # if there's hardly been any progress in the last 30 steps
        #     print("converged 2")
        #     break
        # if new_objective > init_objective * protect_thres:
        #     prot_break = True
        #     print("converged 3")
        #     break

        it = lm[nstep-1]
        itsi = it.s = delta_x
        ityi = it.y = delta_gx
        ys = torch.bmm(ityi.view(bsz, 1, -1), itsi.view(bsz, -1, 1)).squeeze(-1)
        sBs = torch.bmm(itsi.view(bsz, 1, -1), Bs.view(bsz, -1, 1)).squeeze(-1)

        dc = 0.2 # damping
        if (ys<dc*sBs).sum() > 0:
            num_fails_total += (ys<=dc*sBs).sum()
            num_fails += 1
            print("Damping", num_fails, num_fails_total)
            damping = True
            if damping:
                theta = torch.ones_like(ys)
                theta[ys<dc*sBs] = (((1 - dc) * sBs)/torch.clamp(sBs - ys, 1e-14, 100)) [ys<dc*sBs]
                ityi = theta * ityi + (1 - theta) * Bs
                ys = torch.bmm(ityi.view(bsz, 1, -1), itsi.view(bsz, -1, 1)).squeeze(-1)

        # For the limited memory version, uncomment the second line
        # ipdb.set_trace()
        bound = min(lbfgs_mem,nstep)

        # Compute scalars ys and yy:
        yy = torch.bmm(ityi.view(bsz, 1, -1), ityi.view(bsz, -1, 1)).squeeze(-1)
        update = -gx
        # it.y = ityi
        it.ys = ys
        j = nstep
        for i in range(0, bound):
            # from later to former
            j = j-1
            it = lm[j]
            it.alpha = torch.bmm(it.s.view(bsz, 1, -1), update.view(bsz, -1, 1)).squeeze(-1) / (it.ys + 1e-8)
            update = update - (it.y * it.alpha)
        # ipdb.set_trace()
        update = update * (ys/(yy + 1e-8))

        for i in range(0, bound):
            it = lm[j]
            beta = torch.bmm(it.y.view(bsz, 1, -1), update.view(bsz, -1, 1)).squeeze(-1)
            beta = beta /(it.ys + 1e-8)
            update = update + (it.s * (it.alpha - beta))
            # from former to later
            j = j+1

        dyn_res = dyn_fn(x_est) if dyn_fn is not None else 0
        cost = cost_fn(x_est) if cost_fn is not None else 0
        print(nstep, torch.norm(dyn_res).item(), torch.norm(cost).item(), new2_objective, merit.mean().item(), torch.norm(delta_x).item(), torch.norm(x_est).item(), torch.norm(gx).item(), torch.norm(update).item(), stepsz)
        # ipdb.set_trace()
    Us, VTs = None, None
    return {"result": x_est,
            "nstep": nstep,
            "tnstep": tnstep,
            "lowest_step": lowest_step,
            "diff": torch.norm(lowest_gx).item(),
            "diff_detail": torch.norm(lowest_gx, dim=1),
            "prot_break": prot_break,
            "trace": trace,
            "new_trace": new_trace,
            "eps": eps,
            "threshold": threshold,
            "gx": lowest_gx}

def Newton_AL(g, meritfn, dyn_fn, cost_fn, x0, z, threshold, eps, ls=False):
    bsz, total_hsize, n_elem = x0.size() # (bsz, T, xd+ud)
    dev = x0.device

    x_est = x0           # (bsz, 2d, L')
    gx = g(x_est, z)  # (bsz, 2d, L')
    cost = cost_fn(x_est) 
    dyn_res = dyn_fn(x_est)
    tnstep = 0

    # Solve for newton steps on the augmented lagrangian
    nstep = 0
    prot_break = False
    lowest_gx = gx
    lowest_step = 0
    print(nstep, torch.norm(dyn_res).item(), torch.norm(cost).item(), torch.norm(gx).item())
    while torch.norm(gx).item() > threshold and nstep < 4:
        nstep += 1
        
        # Compute the hessian and gradient of the augmented lagrangian
        merit = meritfn(x_est, grad=True).mean()
        grad = torch.autograd.grad(merit, x_est)[0]
        meritfn_mean = lambda x: meritfn(x, grad=True).mean()
        Hess = hessian(meritfn_mean, (x_est))
        # Hess = Hess + torch.eye(total_hsize, device=dev).unsqueeze(0).expand(bsz, total_hsize, total_hsize) * eps
        
        # Solve for the newton step
        stepsz = 0
        reg = 0#1e-5
        Hess = Hess.reshape(10*3,10*3)
        eye = torch.eye(10*3, device=dev)
        # while stepsz < 1e-5:
        update = -torch.linalg.solve(Hess+reg*eye, grad.reshape(-1)).reshape(1,10,3)
        if ls:
            stepsz = 1
            new2_objective = meritfn(x_est + stepsz * update).mean().item()
            while new2_objective > merit.mean().item():
                stepsz *= 0.5
                new2_objective = meritfn(x_est + stepsz * update).mean().item()
                # if stepsz < 1e-8:
                #     break
        # if torch.isnan(update).sum() > 0:
        #     ipdb.set_trace()
        else:
            x_est = x_est + update
            new2_objective = meritfn(x_est).mean().item()
        # reg *= 10
        x_est = x_est + stepsz * update
        gx = g(x_est, z)
        cost = cost_fn(x_est)
        dyn_res = dyn_fn(x_est)
        print(nstep, torch.norm(dyn_res).item(), torch.norm(cost).item(), new2_objective, torch.norm(gx).item(), torch.norm(update).item(), stepsz)
        # ipdb.set_trace()
        
    Us, VTs = None, None
    return {"result": x_est,
            "nstep": nstep,
            "tnstep": tnstep,
            "lowest_step": lowest_step,
            "diff": torch.norm(lowest_gx).item(),
            "diff_detail": torch.norm(lowest_gx, dim=1),
            "prot_break": prot_break,
            "eps": eps,
            "threshold": threshold,
            "gx": gx}

class NewtonAL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, meritfn, dyn_fn, cost_fn, merit_hess, xi, x0, z, lam, rho, Q, q, Hess, threshold, eps, ls, verbose):
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
        if verbose:
            # print(nstep, torch.norm(dyn_res).item(), torch.norm(cost).item(), torch.norm(gx).item())
            print(dyn_res.view(bsz, -1).norm(dim=-1).mean().item(), (cost).mean().item(), torch.norm(gx).item())
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
                Hess, Hess_clip = merit_hess(x_est)
                # Hess = hessian(meritfn_mean, (x_est.reshape(bsz, -1)), vectorize=True)
                # Hess = vmap(hessian(meritfn_mean))(x_est.reshape(bsz, -1), Q_tr, q_tr, lam, x0, rho_new)
            # Hess = Hess + torch.eye(total_hsize, device=dev).unsqueeze(0).expand(bsz, total_hsize, total_hsize) * eps
            # ipdb.set_trace()
            # Solve for the newton step
            stepsz = 0
            reg = 0#reg*10*(1-status) + status*1e-5
            # Hess = Hess.reshape(bsz, T*3,T*3)
            # eye = torch.eye(T*3, device=dev).to(Hess).unsqueeze(0).expand(bsz, T*3, T*3)
            # while stepsz < 1e-5:
            # Hess_clip = Hess = torch.eye(Hess.shape[-1], device=Hess.device, dtype=Hess.dtype).repeat(bsz, 1, 1) 
            if not cholesky_fail:
                U, info = torch.linalg.cholesky_ex(Hess)
                update = -torch.cholesky_solve(grad.reshape(bsz, -1, 1), U).reshape(bsz,T,3)
            else:
                U = Hess_clip
                try:
                    # update = -grad.reshape(bsz, T, 3) #
                    update = -torch.linalg.solve(Hess, grad.reshape(bsz, -1)).reshape(bsz,T,3)
                    # update = torch.round(update, decimals=2)
                except:
                    ipdb.set_trace()
            if ls:
                x_est1, new2_objective, stepsz, status = line_search_newton(update, x_est, meritfnQ, merit)
            else:
                x_est1 = x_est + update
                new2_objective = meritfnQ(x_est).mean().item()
            # reg *= 10
            # x_est = x_est + stepsz * update
            # gx = gQ(x_est, z)
            # cost = cost_fnQ(x_est)
            dyn_res = dyn_fn(x_est1)
            # update_norm = update.norm().item()
            cost = cost_fnQ(x_est1) 
            new_dyn_res = torch.norm(dyn_res).item()
            if verbose:
                # print(nstep, torch.norm(dyn_res).item(), torch.norm(cost).item(), new2_objective, torch.norm(gx).item(), torch.norm(update).item(), stepsz)
                print(nstep, (dyn_res.view(bsz, -1).norm(dim=-1)).mean().item(), (cost).mean().item(), torch.norm(update).item(), new2_objective, stepsz)
            # if nstep == 3:
            #     ipdb.set_trace()
            ## exit creteria
            if abs(old_dyn_res- new_dyn_res)/new_dyn_res < 1e-3 or new_dyn_res < 1e-3:
                break
                
            old_dyn_res = new_dyn_res
            x_est = x_est1
            # rho_new = rho*status[:,None] + rho_new/2*(1-status[:,None])#min(, rho_init*100)
        # print(nstep)
        ctx.save_for_backward(Hess_clip, U, x_est, cholesky_fail)
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

        return None, None, None, None, None, None, None, None, None, None, Q_grad.transpose(0,1), q_grad.transpose(0,1), None, None, None, None, None

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
    new2_objective = vmap(meritfnQ)(x_next)
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
                    
def check_fd_grads(fn, x, eps=1e-8):
    x_shape = x.shape
    x = x.reshape(-1)
    x = x.clone().detach().requires_grad_(True)
    # ipdb.set_trace()
    y = fn(x.view(x_shape))
    # ipdb.set_trace()
    y = y.norm()
    # ipdb.set_trace()
    grad = torch.autograd.grad(y, x)[0]
    J = torch.zeros(grad.shape).to(x)
    for i in range(x.nelement()):
        x1 = x.clone().detach()
        x1[i] += eps
        y1 = fn(x1.view(x_shape))
        y1 = y1.norm()
        x2 = x.clone().detach()
        x2[i] -= eps
        y2 = fn(x2.view(x_shape))
        y2 = y2.norm()
        J[i] = (y1 - y2) / (2*eps)
        # ipdb.set_trace()
    print(torch.norm(J-grad)/torch.norm(grad), torch.norm(J), torch.norm(grad), torch.norm(J-grad))
    ipdb.set_trace()
    return grad, J

def check_grads(fn, x, eps=1e-8):
    # perform gradient descent or adam updates and check if the loss decreases
    x_shape = x.shape
    x = x.reshape(-1)
    # define optimizer 
    optimizer = torch.optim.Adam([x], lr=0.1)
    # optimizer = torch.optim.SGD([x], lr=0.1)

    for i in range(100):
        optimizer.zero_grad()
        y = fn(x.view(x_shape))
        y = y.norm()
        y.backward()
        optimizer.step()
        print(y.item())
    
    return x

class CholeskySolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, b):
        # don't crash training if cholesky decomp fails
        U, info = torch.linalg.cholesky_ex(H)

        if torch.any(info):
            ctx.failed = True
            return torch.zeros_like(b)

        xs = torch.cholesky_solve(b, U)
        ctx.save_for_backward(U, xs)
        ctx.failed = False

        return xs

    @staticmethod
    def backward(ctx, grad_x):
        if ctx.failed:
            return None, None

        U, xs = ctx.saved_tensors
        dz = torch.cholesky_solve(grad_x, U)
        dH = -torch.matmul(xs, dz.transpose(-1,-2))

        return dH, dz
    
def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0): 
    ### TODO : Parallelize this search!
    ite = 0
    phi_a0 = phi(alpha0)    # First do an update with step size 1
    mask = (phi_a0 > phi0 + c1*alpha0*derphi0).float()
    if torch.sum(mask)==0:
        return alpha0, phi_a0, ite

    # Otherwise, compute the minimizer of a quadratic interpolant
    alpha1 = -((derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0))*mask + alpha0*(1-mask)
    phi_a1 = phi(alpha1)

    # Otherwise loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.
    while alpha1.min() > amin:       # we are assuming alpha>0 is a descent direction
        factor = alpha0**2 * alpha1**2 * (alpha1-alpha0)
        a = alpha0**2 * (phi_a1 - phi0 - derphi0*alpha1) - \
            alpha1**2 * (phi_a0 - phi0 - derphi0*alpha0)
        a = a / factor
        b = -alpha0**3 * (phi_a1 - phi0 - derphi0*alpha1) + \
            alpha1**3 * (phi_a0 - phi0 - derphi0*alpha0)
        b = b / factor

        alpha2 = (-b + torch.sqrt(torch.abs(b**2 - 3 * a * derphi0))) / (3.0*a)
        alpha2 = alpha2*mask + alpha1*(1-mask)
        phi_a2 = phi(alpha2)
        ite += 1
        mask = (phi_a2 > phi0 + c1*alpha2*derphi0).float()
        if torch.sum(mask)==0:
            return alpha2.item(), phi_a2, ite
        
    
        # if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
        #     alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2
    mask = alpha1 < amin
    alpha1 = (~mask)*alpha1
    print("unconverged line search")
    # Failed to find a suitable step length
    return alpha1.item(), phi_a1, ite

def scalar_search_armijo2(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0):
    ite = 0
    phi_a0 = phi(alpha0)    # First do an update with step size 1
    mask = phi_a0 > phi0 + c1*alpha0*derphi0
    if torch.sum(mask)==0:
        return alpha0, phi_a0, ite

    alpha1 = mask*alpha0/2.0 + (~mask)*alpha0
    alpha2 = alpha1
    phi_a1 = phi(alpha1)

    while torch.min(alpha1) > amin:       # we are assuming alpha>0 is a descent direction
        phi_a2 = phi(alpha2)
        ite += 1
        mask = phi_a2 > phi0 + c1*alpha2*derphi0
        if torch.sum(mask)==0:
            return alpha2, phi_a2, ite

        alpha2 = mask*alpha1/2.0 + (~mask)*alpha1

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2
    mask = alpha1 < amin
    alpha1 = (~mask)*alpha1

    # Failed to find a suitable step length
    return alpha1, phi_a1, ite

def line_search(update, x0, g0, g, meritfn, nstep=0, on=True):
    """
    `update` is the propsoed direction of update.

    Code adapted from scipy.
    """
    bsz = x0.size(0)
    tmp_s = [0]
    tmp_g0 = [g0]
    tmp_phi = [meritfn(x0)]
    # tmp_phi = [torch.norm(g0)**2]
    s_norm = torch.norm(x0) / torch.norm(update)

    def phi(s, store=True):
        if s == tmp_s[0]:
            return tmp_phi[0]    # If the step size is so small... just return something

        x_est = x0 + s * update
        # g0_new = g(x_est)
        phi_new = meritfn(x_est)#
        # phi_new = _safe_norm(g0_new)**2
        if store:
            tmp_s[0] = s
            # tmp_g0[0] = g0_new
            tmp_phi[0] = phi_new
        return phi_new

    if on:
        derphi = (update.view(bsz, -1)* g0.view(bsz,-1)).sum(dim=1)
        # derphi0 = -tmp_phi[0]
        # derphi = derphi.mean()
        s, phi1, ite = scalar_search_armijo(phi, tmp_phi[0], derphi, amin=1e-12)
    if (not on) or s is None:
        s = 1.0
        ite = 0
        ipdb.set_trace()
    x_est = x0 + s * update
    # if s == tmp_s[0]:
    #     g0_new = tmp_g0[0]
    # else:
    g0_new = g(x_est)
    if (not on) or s is None:
        tmp_phi[-1] = meritfn(x_est)
    return x_est, g0_new, x_est - x0, g0_new - g0, ite, s, tmp_phi[-1]

