import numpy as np
import torch
from enum import Enum
# from block import block
import ipdb


INACC_ERR = """
--------
qpth warning: Returning an inaccurate and potentially incorrect solutino.

Some residual is large.
Your problem may be infeasible or difficult.

You can try using the CVXPY solver to see if your problem is feasible
and you can use the verbose option to check the convergence status of
our solver while increasing the number of iterations.

Advanced users:
You can also try to enable iterative refinement in the solver:
https://github.com/locuslab/qpth/issues/6
--------
"""


class KKTSolvers(Enum):
    QR = 1

def forward(K, Didx, Q, p, G, GT, h, A, AT, b, dyn_res, cost_grad=None,
            eps=1e-12, verbose=0, notImprovedLim=3, maxIter=20):
    '''
    A primal dual interior point method to solve the sparse QP given by the kkt system in Ki
    '''
    # verbose = 1
    nineq, nz = G.shape[1], G.shape[2]
    neq = A.shape[1]
    nBatch = Q.shape[0]
    sizes = (nineq, neq, nz)

    solver = KKTSolvers.QR

    KKTeps = 1e-7  # For the regularized KKT matrix.
    Zidx, Sidx = Didx

    # Find initial values
    if solver == KKTSolvers.QR:
        # Di = torch.LongTensor([range(nineq), range(nineq)]).type_as(Qi)
        # Dv = torch.ones(nBatch, nineq).type_as(Qv)
        Sv = torch.ones(nBatch, nineq).to(Q)
        Zv = torch.ones(nBatch, nineq).to(Q)
        # Dsz = torch.Size([nineq, nineq])
        K[:, Sidx[0], Sidx[1]] = Sv
        K[:, Zidx[0], Zidx[1]] = Zv
        I = torch.ones(nBatch, K.shape[1]).to(Q)
        I[:, nz + nineq:] *= -1
        Ktilde = K + KKTeps * torch.diag_embed(I)
        # K, Didx = cat_kkt(Ki, Ki_cat_idx, Q, G, GT, A, AT, Sv, Zv, sizes, 0.0)
        # Ktilde, Didxtilde = cat_kkt(Ki, Ki_cat_idx, Q, G, GT, A, AT, Sv, Zv, sizes, KKTeps)
        # assert torch.norm((Didx - Didxtilde).float()) == 0.0
        # Z1 = torch.zeros(nBatch, nineq, nz).type_as(Q)
        # I1 = torch.eye(nineq).to(Q)[None].repeat(nBatch, 1, 1)
        # K1 = Ktilde1 = torch.cat([
        #                             torch.cat([Q, torch.zeros(nBatch, nz, nineq).to(Q), GT, AT], dim=-1),
        #                             torch.cat([Z1, I1, I1, torch.zeros(nBatch, nineq, neq).to(Q)], dim=-1),
        #                             torch.cat([G, I1, torch.zeros(nBatch, nineq, neq+nineq).type_as(Q)], dim=-1),
        #                             torch.cat([A, torch.zeros(nBatch, neq, neq+nineq*2).type_as(Q)], dim=-1),
        #                         ], dim=-2)
        # ipdb.set_trace()
        x, s, z, y = solve_kkt(K, Ktilde, p, torch.zeros(nBatch, nineq).to(p),
                               -h, -b if b is not None else None)
        # ipdb.set_trace()
    else:
        assert False

    M = torch.min(s, 1)[0][:,None].repeat(1, nineq)
    I = M < 0
    s[I] -= M[I] - 1

    M = torch.min(z, 1)[0][:,None].repeat(1, nineq)
    I = M < 0
    z[I] -= M[I] - 1

    best = {'resids': None, 'x': None, 'z': None, 's': None, 'y': None, 'K': None}
    nNotImproved = 0

    for i in range(maxIter):
        # affine scaling direction
        if cost_grad is None:
            rx = ((AT.bmm(y.unsqueeze(-1)).squeeze(-1) if neq > 0 else 0.) +
                GT.bmm(z.unsqueeze(-1)).squeeze(-1) + 
                Q.bmm(x.unsqueeze(-1)).squeeze(-1) + p)
        else:
            rx = ((AT.bmm(y.unsqueeze(-1)).squeeze(-1) if neq > 0 else 0.) +
                GT.bmm(z.unsqueeze(-1)).squeeze(-1) + cost_grad(x))
        rs = s*z
        rz = (G.bmm(x.unsqueeze(-1)).squeeze(-1) + s - h)
        ry = dyn_res(x)#(A.bmm(x.unsqueeze(-1)).squeeze(-1) - b)
        mu = torch.abs((s * z).sum(1).squeeze() / nineq)
        z_resid = torch.norm(rz, 2, 1).squeeze()
        y_resid = torch.norm(ry, 2, 1).squeeze() if neq > 0 else 0
        pri_resid = y_resid + z_resid
        dual_resid = torch.norm(rx, 2, 1).squeeze()
        resids = pri_resid + dual_resid + nineq * mu
        # D = z / s
        # ipdb.set_trace()
        # K.value[:,Didx[0]] = z
        # K.value[:,Didx[1]] = s
        # Ktilde.value[:, Didx[0]] = z + KKTeps
        # Ktilde.value[:, Didx[1]] = s + KKTeps
        K[:,Zidx[0], Zidx[1]] = z
        K[:,Sidx[0], Sidx[1]] = s
        Ktilde[:, Zidx[0], Zidx[1]] = z + KKTeps
        Ktilde[:, Sidx[0], Sidx[1]] = s

        if verbose == 1:
            print('iter: {}, pri_resid: {:.5e}, dual_resid: {:.5e}, mu: {:.5e}'.format(
                i, pri_resid.mean(), dual_resid.mean(), mu.mean()))
        if best['resids'] is None:
            best['resids'] = resids
            best['x'] = x.clone()
            best['z'] = z.clone()
            best['s'] = s.clone()
            best['y'] = y.clone() if y is not None else None
            best['K'] = K.clone()
            nNotImproved = 0
        else:
            I = resids < best['resids']
            if I.sum() > 0:
                nNotImproved = 0
            else:
                nNotImproved += 1
                # KKTeps /= 1000
            I_nz = I.repeat(nz, 1).t()
            I_nineq = I.repeat(nineq, 1).t()
            I_K = I.repeat(K.shape[2], K.shape[1], 1).transpose(0,2)
            best['resids'][I] = resids[I]
            best['x'][I_nz] = x[I_nz]
            best['z'][I_nineq] = z[I_nineq]
            best['s'][I_nineq] = s[I_nineq]
            best['K'][I_K] = K[I_K]
            if neq > 0:
                I_neq = I.repeat(neq, 1).t()
                best['y'][I_neq] = y[I_neq]
        if nNotImproved == notImprovedLim or best['resids'].max() < eps or mu.min() > 1e32:
            if best['resids'].max() > 1. and verbose >= 0:
                print(INACC_ERR)
            # ipdb.set_trace()
            return best['x'], best['y'], best['z'], best['s'], best['K']

        if solver == KKTSolvers.QR:
            # ipdb.set_trace()
            dx_aff, ds_aff, dz_aff, dy_aff = solve_kkt(
                K, Ktilde, rx, rs, rz, ry)
            # ipdb.set_trace()
        else:
            assert False

        # compute centering directions
        alpha = torch.min(torch.min(get_step(z, dz_aff),
                                    get_step(s, ds_aff)),
                          torch.ones(nBatch).type_as(Q))
        alpha_nineq = alpha.repeat(nineq, 1).t()
        t1 = s + alpha_nineq * ds_aff
        t2 = z + alpha_nineq * dz_aff
        t3 = torch.sum(t1 * t2, 1).squeeze()
        t4 = torch.sum(s * z, 1).squeeze()
        sig = (t3 / t4)**3

        rx = torch.zeros(nBatch, nz).type_as(Q)
        rs = ((-mu * sig).repeat(nineq, 1).t() + ds_aff * dz_aff)# / s
        rz = torch.zeros(nBatch, nineq).type_as(Q)
        ry = torch.zeros(nBatch, neq).type_as(Q)

        if solver == KKTSolvers.QR:
            dx_cor, ds_cor, dz_cor, dy_cor = solve_kkt(
                K, Ktilde, rx, rs, rz, ry)
            # ipdb.set_trace()
        else:
            assert False

        dx = dx_aff + dx_cor
        ds = ds_aff + ds_cor
        dz = dz_aff + dz_cor
        dy = dy_aff + dy_cor if neq > 0 else None
        alpha = torch.min(0.999 * torch.min(get_step(z, dz),
                                            get_step(s, ds)),
                          torch.ones(nBatch).type_as(Q))

        alpha_nineq = alpha.repeat(nineq, 1).t()
        alpha_neq = alpha.repeat(neq, 1).t() if neq > 0 else None
        alpha_nz = alpha.repeat(nz, 1).t()

        x += alpha_nz * dx
        s += alpha_nineq * ds
        z += alpha_nineq * dz
        y = y + alpha_neq * dy if neq > 0 else None
        # ipdb.set_trace()

    if best['resids'].max() > 1. and verbose >= 0:
        print(INACC_ERR)
    return best['x'], best['y'], best['z'], best['s'], best['K']


def get_step(v, dv):
    # nBatch = v.size(0)
    # ipdb.set_trace()
    a = -v / dv
    a[dv == 0] = 1.0
    a[dv > 0] = max(1.0, a.max())
    return a.min(1)[0].squeeze()

def solve_kkt(K, Ktilde, 
              rx, rs, rz, ry, niter=1):
    nBatch = K.shape[0]
    nz = rx.size(1)
    nineq = rz.size(1)
    neq = ry.size(1)

    r = -torch.cat((rx, rs, rz, ry), 1)

    l = r.clone().detach()#torch.zeros_like(r) + r#torch.spbqrfactsolve(*([r] + Ktilde))
    # solver_ctx.factor(K) # need to check matrix type
    # ipdb.set_trace()
    K_LU = torch.linalg.lu_factor(Ktilde)
    l = torch.linalg.lu_solve(*K_LU, l.unsqueeze(-1)).squeeze(-1)

    # Iterative refinement
    res = r - K.bmm(l.unsqueeze(-1)).squeeze(-1)
    for k in range(niter):
        # d = torch.spbqrfactsolve(*([res] + Ktilde))
        d = res.clone()#torch.zeros_like(res)
        # solver_ctx.solve(d)
        d = torch.linalg.lu_solve(*K_LU, d.unsqueeze(-1)).squeeze(-1)
        l = l + d
        # ipdb.set_trace()
        res = r - K.bmm(l.unsqueeze(-1)).squeeze(-1)

    # print("solved")
    solx = l[:, :nz]
    sols = l[:, nz:nz + nineq]
    solz = l[:, nz + nineq:nz + 2 * nineq]
    soly = l[:, nz + 2 * nineq:nz + 2 * nineq + neq]

    return solx, sols, solz, soly#s, solz, soly
