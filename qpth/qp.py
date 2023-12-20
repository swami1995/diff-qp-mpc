import torch
from torch.autograd import Function

from .util import bger, expandParam, extract_nBatch
from . import solvers
from .solvers.pdipm import batch as pdipm_b
from .solvers.pdipm import spbatch as pdipm_spb
from .solvers.pdipm import SparseStructure as ss
from .solvers.pdipm import spbatch_LU as pdipm_spb_LU
from .solvers.pdipm import batch_LU as pdipm_b_LU
# from .solvers.pdipm import single as pdipm_s

from enum import Enum
from qpth.extlib.cusolver_lu_solver import CusolverLUSolver
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


class SpQPFunction(Function):
    def __init__(self, Qi, Qsz, Gi, Gsz, Ai, Asz,
                 eps=1e-12, verbose=0, notImprovedLim=3, maxIter=20):
        self.Qi, self.Qsz = Qi, Qsz
        self.Gi, self.Gsz = Gi, Gsz
        self.Ai, self.Asz = Ai, Asz

        self.eps = eps
        self.verbose = verbose
        self.notImprovedLim = notImprovedLim
        self.maxIter = maxIter

        self.nineq, self.nz = Gsz
        self.neq, _ = Asz

    def forward(self, Qv, p, Gv, h, Av, b):
        self.nBatch = Qv.size(0)

        zhats, self.nus, self.lams, self.slacks, K = pdipm_spb.forward(
            self.Qi, Qv, self.Qsz, p, self.Gi, Gv, self.Gsz, h,
            self.Ai, Av, self.Asz, b, self.eps, self.verbose,
            self.notImprovedLim, self.maxIter)

        self.save_for_backward(zhats, Qv, p, Gv, h, Av, b)
        return zhats

    def backward(self, dl_dzhat):
        zhats, Qv, p, Gv, h, Av, b = self.saved_tensors

        Di = type(self.Qi)([range(self.nineq), range(self.nineq)])
        Dv = self.lams / self.slacks
        Dsz = torch.Size([self.nineq, self.nineq])
        dx, _, dlam, dnu = pdipm_spb.solve_kkt(
            self.Qi, Qv, self.Qsz, Di, Dv, Dsz,
            self.Gi, Gv, self.Gsz,
            self.Ai, Av, self.Asz, dl_dzhat,
            type(p)(self.nBatch, self.nineq).zero_(),
            type(p)(self.nBatch, self.nineq).zero_(),
            type(p)(self.nBatch, self.neq).zero_())

        dps = dx

        dGs = bger(dlam, zhats) + bger(self.lams, dx)
        GM = torch.cuda.sparse.DoubleTensor(
            self.Gi, Gv[0].clone().fill_(1.0), self.Gsz
        ).to_dense().byte().expand_as(dGs)
        dGs = dGs[GM].view_as(Gv)

        dhs = -dlam

        dAs = bger(dnu, zhats) + bger(self.nus, dx)
        AM = torch.cuda.sparse.DoubleTensor(
            self.Ai, Av[0].clone().fill_(1.0), self.Asz
        ).to_dense().byte().expand_as(dAs)
        dAs = dAs[AM].view_as(Av)

        dbs = -dnu

        dQs = 0.5 * (bger(dx, zhats) + bger(zhats, dx))
        QM = torch.cuda.sparse.DoubleTensor(
            self.Qi, Qv[0].clone().fill_(1.0), self.Qsz
        ).to_dense().byte().expand_as(dQs)
        dQs = dQs[QM].view_as(Qv)

        grads = (dQs, dps, dGs, dhs, dAs, dbs)

        return grads



def SparseQPFunction(Qi, Gi, Ai, bsz = 1,
                 eps=1e-12, verbose=0, notImprovedLim=3, maxIter=20):
    
    # def __init__(self, ):
    #     super(SparseQPFunction, self).__init__()
        # Qi = Qi
        # Gi = Gi
        # Ai = Ai
    GTi = Gi.transpose()
    ATi = Ai.transpose()
    Qcol_idx, Qrw_ptr = Qi.col_ind, Qi.row_ptr
    Gcol_idx, Grw_ptr = Gi.col_ind, Gi.row_ptr
    Acol_idx, Arw_ptr = Ai.col_ind, Ai.row_ptr
    GTcol_idx, GTrw_ptr = GTi.col_ind, GTi.row_ptr
    ATcol_idx, ATrw_ptr = ATi.col_ind, ATi.row_ptr

    def compute_KKT_i():
        Kcol_idx = []
        Krow_idx = []
        Krw_ptr = [0]
        Ki_cat_idx = []
        Ki_cat_idx1 = []
        for i in range(len(Qrw_ptr)-1):
            Kcol_idx.append(Qcol_idx[Qrw_ptr[i]:Qrw_ptr[i+1]])
            Kcol_idx.append(GTcol_idx[GTrw_ptr[i]:GTrw_ptr[i+1]] + Qi.num_cols + Gi.num_rows)
            Kcol_idx.append(ATcol_idx[ATrw_ptr[i]:ATrw_ptr[i+1]] + Qi.num_cols + Gi.num_rows + Gi.num_rows)

            num_elems = Kcol_idx[-1].shape[-1] + Kcol_idx[-2].shape[-1] + Kcol_idx[-3].shape[-1]
            Krow_idx.append(torch.Tensor([i]*num_elems))
            Krw_ptr.append(Krw_ptr[-1] + num_elems)

            Ki_cat_idx1.append(torch.arange(Qrw_ptr[i],Qrw_ptr[i+1]))
            Ki_cat_idx1.append(torch.arange(GTrw_ptr[i], GTrw_ptr[i+1]) + Qrw_ptr[-1])
            Ki_cat_idx1.append(torch.arange(ATrw_ptr[i], ATrw_ptr[i+1]) + Qrw_ptr[-1] + GTrw_ptr[-1])
        
        Ki_cat_idx2 = []
        for i in range(len(Grw_ptr)-1):
            Kcol_idx.append(torch.Tensor([Qi.num_cols + i]))
            Kcol_idx.append(torch.Tensor([Qi.num_cols + Gi.num_rows + i]))

            num_elems = 2
            Krow_idx.append(torch.Tensor([i+Qi.num_rows]*num_elems))
            Krw_ptr.append(Krw_ptr[-1] + num_elems)

            Ki_cat_idx2.append(torch.Tensor([i]))
            Ki_cat_idx2.append(torch.Tensor([i]) + Gi.num_rows)

        Ki_cat_idx3 = []
        for i in range(len(Grw_ptr)-1):
            Kcol_idx.append(Gcol_idx[Grw_ptr[i]:Grw_ptr[i+1]])   
            Kcol_idx.append(torch.Tensor([Gi.num_cols + i]))
            Kcol_idx.append(torch.Tensor([Gi.num_cols + Gi.num_rows + i]))

            num_elems = Kcol_idx[-1].shape[-1] + Kcol_idx[-2].shape[-1] + Kcol_idx[-3].shape[-1]
            Krow_idx.append(torch.Tensor([i+Qi.num_rows+Gi.num_rows]*num_elems))
            Krw_ptr.append(Krw_ptr[-1] + num_elems)

            Ki_cat_idx3.append(torch.arange(Grw_ptr[i], Grw_ptr[i+1]))
            Ki_cat_idx3.append(torch.Tensor([i]) + Grw_ptr[-1])
            Ki_cat_idx3.append(torch.Tensor([i]) + Grw_ptr[-1] + Gi.num_rows)

        Ki_cat_idx4 = []
        for i in range(len(Arw_ptr)-1):
            Kcol_idx.append(Acol_idx[Arw_ptr[i]:Arw_ptr[i+1]])
            Kcol_idx.append(torch.Tensor([Ai.num_cols + Gi.num_rows + Gi.num_rows + i]))

            num_elems = Kcol_idx[-1].shape[-1] + Kcol_idx[-2].shape[-1]
            Krow_idx.append(torch.Tensor([i+Qi.num_rows+Gi.num_rows]*num_elems))
            Krw_ptr.append(Krw_ptr[-1] + num_elems)

            Ki_cat_idx4.append(torch.arange(Arw_ptr[i], Arw_ptr[i+1]))
            Ki_cat_idx4.append(torch.Tensor([i]) + Arw_ptr[-1])

        Ki_cat_idx.append(torch.cat(Ki_cat_idx1).long())
        Ki_cat_idx.append(torch.cat(Ki_cat_idx2).long())
        Ki_cat_idx.append(torch.cat(Ki_cat_idx3).long())
        Ki_cat_idx.append(torch.cat(Ki_cat_idx4).long())
        Kcol_idx = torch.cat(Kcol_idx).int()
        Krw_ptr = torch.Tensor(Krw_ptr).int()
        Krow_idx = torch.cat(Krow_idx).int()
        # ipdb.set_trace()
        return Kcol_idx, Krw_ptr, Ki_cat_idx#Krow_idx, 

    
    Kcol_idx, Krw_ptr, Ki_cat_idx = compute_KKT_i()
    num_rows = num_cols = Qi.num_cols + Gi.num_rows*2 + Ai.num_rows
    Ki = ss.SparseStructure(Krw_ptr, Kcol_idx, num_rows=num_rows, num_cols=num_cols)
    Ki_cat_idx = Ki_cat_idx

    eps = eps
    verbose = verbose
    notImprovedLim = notImprovedLim
    maxIter = maxIter

    nineq, nz = Gi.size()
    neq, _ = Ai.size()
    # ipdb.set_trace()
    _solver_contexts: List[CusolverLUSolver] = [
        CusolverLUSolver(
            bsz,
            Ki.shape[1],
            Ki.row_ptr.cuda().int(),
            Ki.col_ind.cuda().int(),
        )
        for _ in range(10)
    ]
    
    def make_csrs(Qv, Gv, Av):
        # ipdb.set_trace()
        nbatch = Qv.size(0)
        Q = ss.SparseStructure(Qrw_ptr[None].repeat(nbatch, 1), Qcol_idx[None].repeat(nbatch, 1), Qv, Qi.num_rows,Qi.num_cols).double().cuda()
        G = ss.SparseStructure(Grw_ptr[None].repeat(nbatch, 1), Gcol_idx[None].repeat(nbatch, 1), Gv, Gi.num_rows,Gi.num_cols).double().cuda()
        A = ss.SparseStructure(Arw_ptr[None].repeat(nbatch, 1), Acol_idx[None].repeat(nbatch, 1), Av, Ai.num_rows,Ai.num_cols).double().cuda()
        GT = G.transpose()
        AT = A.transpose()
        return Q, G, GT, A, AT
        
    class Solver(Function):
        @staticmethod
        def forward(ctx, Qv, p, Gv, h, Av, b):
            nBatch = Qv.size(0)
            Q, G, GT, A, AT = make_csrs(Qv, Gv, Av)
            p = p.double().cuda()
            h = h.double().cuda()
            b = b.double().cuda()
            # ipdb.set_trace()

            zhats, nus, lams, slacks, K, Ktilde = pdipm_spb_LU.forward(
                _solver_contexts[0], Ki, Ki_cat_idx, Q, p, G, GT, h, A, AT, b, eps, verbose,
                notImprovedLim, maxIter)

            ctx.save_for_backward(zhats, nus, lams, K.value, Ktilde.value, K.row_ptr, K.col_ind, Q.value, p, G.value, A.value)
            return zhats

        @staticmethod
        def backward(ctx, dl_dzhat):
            zhats, nus, lams, Kv, Ktildev, Kr, Kc, Qv, p, Gv, Av = ctx.saved_tensors

            # Di = type(Qi)([range(nineq), range(nineq)])
            # Dv = lams / slacks
            # Dsz = torch.Size([nineq, nineq])
            K = ss.SparseStructure(Kr, Kc, Kv)#, num_rows=num_rows, num_cols=num_cols)
            Ktilde = ss.SparseStructure(Kr, Kc, Ktildev)#, num_rows=num_rows, num_cols=num_cols)
            nBatch = Qv.shape[0]

            dx, _, dlam, dnu = pdipm_spb_LU.solve_kkt(
                _solver_contexts[0], K, Ktilde, dl_dzhat,
                type(p)(nBatch, nineq).zero_().to(dl_dzhat),
                type(p)(nBatch, nineq).zero_().to(dl_dzhat),
                type(p)(nBatch, neq).zero_().to(dl_dzhat))

            dps = dx

            dGs = bger(dlam, zhats) + bger(lams, dx)
            GM = torch.sparse_csr_tensor(Gi.row_ptr, Gi.col_ind, Gi.value).to_dense().bool().expand_as(dGs)
            # GM = torch.cuda.sparse.DoubleTensor(
                # Gi, G[0].clone().fill_(1.0), Gsz
            # ).to_dense().byte().expand_as(dGs)
            dGs = dGs[GM].view_as(Gv)

            dhs = -dlam

            dAs = bger(dnu, zhats) + bger(nus, dx)
            # AM = torch.cuda.sparse.DoubleTensor(
            #     Ai, A[0].clone().fill_(1.0), Asz
            # ).to_dense().byte().expand_as(dAs)
            # ipdb.set_trace()
            AM = torch.sparse_csr_tensor(Ai.row_ptr, Ai.col_ind, Ai.value, size=(Ai.num_rows, Ai.num_cols)).to_dense().bool().expand_as(dAs)
            dAs = dAs[AM].view_as(Av)

            dbs = -dnu

            dQs = 0.5 * (bger(dx, zhats) + bger(zhats, dx))
            QM = torch.sparse_csr_tensor(Qi.row_ptr, Qi.col_ind, Qi.value).to_dense().bool().expand_as(dQs)
            dQs = dQs[QM].view_as(Qv)

            # grads = (dQs.cpu(), dps.cpu(), dGs.cpu(), dhs.cpu(), dAs.cpu(), dbs.cpu())
            grads = (dQs, dps, dGs, dhs, dAs, dbs)

            return grads
    return Solver.apply


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

            dx, _, dlam, dnu = pdipm_b_LU.solve_kkt(
                K, K, dl_dzhat,
                torch.zeros(nBatch, nineq).to(p),
                torch.zeros(nBatch, nineq).to(p),
                torch.zeros(nBatch, neq).to(p))

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