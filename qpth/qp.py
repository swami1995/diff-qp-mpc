import torch
from torch.autograd import Function

from .util import bger, expandParam, extract_nBatch
from . import solvers
from .solvers.pdipm import batch as pdipm_b
from .solvers.pdipm import batch_LU as pdipm_b_LU
# from .solvers.pdipm import single as pdipm_s

from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union
import ipdb

class QPSolvers(Enum):
    PDIPM_BATCHED = 1
    CVXPY = 2


def QPFunction(eps=1e-12, verbose=0, notImprovedLim=3,
                 maxIter=20, solver=QPSolvers.PDIPM_BATCHED,
                 check_Q_spd=True):
    class QPFunctionFn(Function):
        @staticmethod
        def forward(ctx, Q_, p_, G_, h_, A_, b_):
            """Solve a batch of QPs.

            This function solves a batch of QPs, each optimizing over
            `nz` variables and having `nineq` inequality constraints
            and `neq` equality constraints.
            The optimization problem for each instance in the batch
            (dropping indexing from the notation) is of the form

                \hat z =   argmin_z 1/2 z^T Q z + p^T z
                        subject to Gz <= h
                                    Az  = b

            where Q \in S^{nz,nz},
                S^{nz,nz} is the set of all positive semi-definite matrices,
                p \in R^{nz}
                G \in R^{nineq,nz}
                h \in R^{nineq}
                A \in R^{neq,nz}
                b \in R^{neq}

            These parameters should all be passed to this function as
            Variable- or Parameter-wrapped Tensors.
            (See torch.autograd.Variable and torch.nn.parameter.Parameter)

            If you want to solve a batch of QPs where `nz`, `nineq` and `neq`
            are the same, but some of the contents differ across the
            minibatch, you can pass in tensors in the standard way
            where the first dimension indicates the batch example.
            This can be done with some or all of the coefficients.

            You do not need to add an extra dimension to coefficients
            that will not change across all of the minibatch examples.
            This function is able to infer such cases.

            If you don't want to use any equality or inequality constraints,
            you can set the appropriate values to:

                e = Variable(torch.Tensor())

            Parameters:
            Q:  A (nBatch, nz, nz) or (nz, nz) Tensor.
            p:  A (nBatch, nz) or (nz) Tensor.
            G:  A (nBatch, nineq, nz) or (nineq, nz) Tensor.
            h:  A (nBatch, nineq) or (nineq) Tensor.
            A:  A (nBatch, neq, nz) or (neq, nz) Tensor.
            b:  A (nBatch, neq) or (neq) Tensor.

            Returns: \hat z: a (nBatch, nz) Tensor.
            """
            nBatch = extract_nBatch(Q_, p_, G_, h_, A_, b_)
            Q, _ = expandParam(Q_, nBatch, 3)
            p, _ = expandParam(p_, nBatch, 2)
            G, _ = expandParam(G_, nBatch, 3)
            h, _ = expandParam(h_, nBatch, 2)
            A, _ = expandParam(A_, nBatch, 3)
            b, _ = expandParam(b_, nBatch, 2)

            if check_Q_spd:
                for i in range(nBatch):
                    e, _ = torch.linalg.eig(Q[i])
                    if not torch.all(e.real > 0):
                        raise RuntimeError('Q is not SPD.')

            _, nineq, nz = G.size()
            neq = A.size(1) if A.nelement() > 0 else 0
            assert(neq > 0 or nineq > 0)
            ctx.neq, ctx.nineq, ctx.nz = neq, nineq, nz

            if solver == QPSolvers.PDIPM_BATCHED:
                ctx.Q_LU, ctx.S_LU, ctx.R = pdipm_b.pre_factor_kkt(Q, G, A)
                zhats, ctx.nus, ctx.lams, ctx.slacks = pdipm_b.forward(
                    Q, p, G, h, A, b, ctx.Q_LU, ctx.S_LU, ctx.R,
                    eps, verbose, notImprovedLim, maxIter)
            elif solver == QPSolvers.CVXPY:
                vals = torch.Tensor(nBatch).type_as(Q)
                zhats = torch.Tensor(nBatch, ctx.nz).type_as(Q)
                lams = torch.Tensor(nBatch, ctx.nineq).type_as(Q)
                nus = torch.Tensor(nBatch, ctx.neq).type_as(Q) \
                    if ctx.neq > 0 else torch.Tensor()
                slacks = torch.Tensor(nBatch, ctx.nineq).type_as(Q)
                for i in range(nBatch):
                    Ai, bi = (A[i], b[i]) if neq > 0 else (None, None)
                    vals[i], zhati, nui, lami, si = solvers.cvxpy.forward_single_np(
                        *[x.cpu().numpy() if x is not None else None
                        for x in (Q[i], p[i], G[i], h[i], Ai, bi)])
                    # if zhati[0] is None:
                    #     import IPython, sys; IPython.embed(); sys.exit(-1)
                    zhats[i] = torch.Tensor(zhati)
                    lams[i] = torch.Tensor(lami)
                    slacks[i] = torch.Tensor(si)
                    if neq > 0:
                        nus[i] = torch.Tensor(nui)

                ctx.vals = vals
                ctx.lams = lams
                ctx.nus = nus
                ctx.slacks = slacks
            else:
                assert False

            ctx.save_for_backward(zhats, Q_, p_, G_, h_, A_, b_)
            return zhats

        @staticmethod
        def backward(ctx, dl_dzhat):
            zhats, Q, p, G, h, A, b = ctx.saved_tensors
            nBatch = extract_nBatch(Q, p, G, h, A, b)
            Q, Q_e = expandParam(Q, nBatch, 3)
            p, p_e = expandParam(p, nBatch, 2)
            G, G_e = expandParam(G, nBatch, 3)
            h, h_e = expandParam(h, nBatch, 2)
            A, A_e = expandParam(A, nBatch, 3)
            b, b_e = expandParam(b, nBatch, 2)

            # neq, nineq, nz = ctx.neq, ctx.nineq, ctx.nz
            neq, nineq = ctx.neq, ctx.nineq


            if solver == QPSolvers.CVXPY:
                ctx.Q_LU, ctx.S_LU, ctx.R = pdipm_b.pre_factor_kkt(Q, G, A)

            # Clamp here to avoid issues coming up when the slacks are too small.
            # TODO: A better fix would be to get lams and slacks from the
            # solver that don't have this issue.
            d = torch.clamp(ctx.lams, min=1e-8) / torch.clamp(ctx.slacks, min=1e-8)

            pdipm_b.factor_kkt(ctx.S_LU, ctx.R, d)
            dx, _, dlam, dnu = pdipm_b.solve_kkt(
                ctx.Q_LU, d, G, A, ctx.S_LU,
                dl_dzhat, torch.zeros(nBatch, nineq).type_as(G),
                torch.zeros(nBatch, nineq).type_as(G),
                torch.zeros(nBatch, neq).type_as(G) if neq > 0 else torch.Tensor())

            dps = dx
            dGs = bger(dlam, zhats) + bger(ctx.lams, dx)
            if G_e:
                dGs = dGs.mean(0)
            dhs = -dlam
            if h_e:
                dhs = dhs.mean(0)
            if neq > 0:
                dAs = bger(dnu, zhats) + bger(ctx.nus, dx)
                dbs = -dnu
                if A_e:
                    dAs = dAs.mean(0)
                if b_e:
                    dbs = dbs.mean(0)
            else:
                dAs, dbs = None, None
            dQs = 0.5 * (bger(dx, zhats) + bger(zhats, dx))
            if Q_e:
                dQs = dQs.mean(0)
            if p_e:
                dps = dps.mean(0)


            grads = (dQs, dps, dGs, dhs, dAs, dbs)

            return grads
    return QPFunctionFn.apply


def DenseQPFunction(bsz = 1,
                 eps=1e-12, verbose=0, notImprovedLim=3, maxIter=20):
    
    eps = eps
    verbose = verbose
    notImprovedLim = notImprovedLim
    maxIter = maxIter

    def preprocess(Q, G, A):
        # ipdb.set_trace()
        nz = Q.size(1)
        neq = A.size(1)
        nineq = G.size(1)
        nbatch = Q.size(0)
        GT = G.transpose(dim0=-2, dim1=-1)
        AT = A.transpose(dim0=-2, dim1=-1)
        Zidx = torch.stack([torch.arange(nz, nz+nineq)]*2, dim=0)
        Sidx = torch.stack([torch.arange(nz, nz+nineq), torch.arange(nz+nineq, nz+2*nineq)]*2, dim=0)
        Didx = (Zidx, Sidx)
        Iin = torch.eye(nineq)[None].to(Q).repeat(nbatch, 1, 1)
        Z1 = torch.zeros(neq, nineq)[None].to(Q).repeat(nbatch, 1, 1)
        Z2 = torch.zeros(nineq, neq)[None].to(Q).repeat(nbatch, 1, 1)
        Z3 = torch.zeros(nineq, nineq)[None].to(Q).repeat(nbatch, 1, 1)
        Z4 = torch.zeros(neq, neq)[None].to(Q).repeat(nbatch, 1, 1)
        K1 = torch.cat([Q, GT*0, GT, AT], dim=-1)
        K2 = torch.cat([G*0, Iin, Iin, Z2], dim=-1)
        K3 = torch.cat([G, Iin, Z3, Z2], dim=-1)
        K4 = torch.cat([A, Z1, Z1, Z4], dim=-1)
        K = torch.cat([K1, K2, K3, K4], dim=-2)#.double()

        return K, GT, AT, Didx
        
    class Solver(Function):
        @staticmethod
        def forward(ctx, Q, p, G, h, A, b):
            nBatch = Q.size(0)
            # Q = Q.double()#.cuda()
            # G = G.double()#.cuda()
            # A = A.double()#.cuda()
            # p = p.double()#.cuda()
            # h = h.double()#.cuda()
            # b = b.double()#.cuda()
            K, GT, AT, Didx = preprocess(Q, G, A)

            zhats, nus, lams, slacks, K = pdipm_b_LU.forward(
                K, Didx, Q, p, G, GT, h, A, AT, b, eps, verbose,
                notImprovedLim, maxIter)

            ctx.save_for_backward(zhats, nus, lams, K, Q, p, G, A)
            return zhats

        @staticmethod
        def backward(ctx, dl_dzhat):
            zhats, nus, lams, K, Q, p, G, A = ctx.saved_tensors

            # Di = type(Qi)([range(nineq), range(nineq)])
            # Dv = lams / slacks
            # Dsz = torch.Size([nineq, nineq])
            nBatch = Q.size(0)
            nineq, neq = G.size(1), A.size(1)
            # print(" dl_dzhat : ", dl_dzhat.isnan().sum())
            dx, _, dlam, dnu = pdipm_b_LU.solve_kkt(
                K, K, dl_dzhat,
                torch.zeros(nBatch, nineq).to(p),
                torch.zeros(nBatch, nineq).to(p),
                torch.zeros(nBatch, neq).to(p))
            # print(" dx, dlam, dnu : ", 
            #     dx.isnan().sum(), dlam.isnan().sum(), dnu.isnan().sum())
            dps = dx

            dGs = bger(dlam, zhats) + bger(lams, dx)

            dhs = -dlam

            dAs = bger(dnu, zhats) + bger(nus, dx)

            dbs = -dnu

            dQs = 0.5 * (bger(dx, zhats) + bger(zhats, dx))
            # ipdb.set_trace()

            grads = (dQs, dps, dGs, dhs, dAs, dbs)

            return grads
    return Solver.apply