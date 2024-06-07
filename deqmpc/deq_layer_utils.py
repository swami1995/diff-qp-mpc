import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import ipdb
from fp_solvers import broyden, anderson

class GradNormLayer(nn.Module):
    def __init__(self, input_size):
        super(GradNormLayer, self).__init__()
        self.input_size = input_size
        self.gradient_moving_avg = torch.zeros(input_size).cuda()#nn.Parameter(torch.zeros(input_size), requires_grad=False)
    
    def forward(self, input):
        return GradNormLayerFunction.apply(input, self.gradient_moving_avg)

class GradNormLayerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gradient_medians):
        ctx.save_for_backward(gradient_medians)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        gradient_moving_avgs = ctx.saved_tensors[0]
        grad_input = grad_output.clone()
        grad_input_flat = grad_input.view(-1, grad_input.size(-1))
        # if gradient_moving_avgs.mean() == 0:
        gradient_moving_avgs = grad_input_flat.abs().mean(dim=0)
        # else:
        #     gradient_moving_avgs[:] = gradient_moving_avgs * 0.9 + grad_input_flat.abs().mean(dim=0) * 0.1
        grad_avg = gradient_moving_avgs.mean()
        grad_input_flat = grad_input_flat * grad_avg / (gradient_moving_avgs[None] + 1e-12)
        # ipdb.set_trace()
        grad_input = grad_input_flat.view(grad_output.size())
        return grad_input, None

class ScaleMultiplyLayer(nn.Module):
    def __init__(self,):
        super(ScaleMultiplyLayer, self).__init__()
    
    def forward(self, input, scale):
        return ScaleMultiplyLayerFunction.apply(input, scale)

class ScaleMultiplyLayerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.save_for_backward(scale, input)
        return input * scale

    @staticmethod
    def backward(ctx, grad_output):
        scale, input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_scale = (grad_output * input)#.sum(dim=-1).unsqueeze(-1)
        return grad_input, grad_scale

def update_scales(policy, trajs, gt_out, init_states, gamma=0.98):
    error0 = (gt_out[:, 1:] - init_states[:, 1:]).abs().median(dim=0).values
    # scale = [error0]
    policy.model.scales[0].data = policy.model.scales[0].data*gamma + (1-gamma)*error0
    for i, traj in enumerate(trajs[:-1]):
        if i >= len(policy.model.scales)-1:
            break
        error = traj[1][:, 1:] - gt_out[:, 1:]
        error = error.abs().median(dim=0).values
        policy.model.scales[i+1].data = policy.model.scales[i+1].data*gamma + (1-gamma)*error

class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(2*dim, 2*dim),
            nn.LayerNorm(dim*2, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(2*dim, dim),
            nn.LayerNorm(dim, eps=1e-3),
            nn.Sigmoid())

        self.res = nn.Sequential(
            nn.Linear(2*dim, 2*dim),
            nn.LayerNorm(dim*2, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(2*dim, dim),
            nn.LayerNorm(dim, eps=1e-3))

    def forward(self, mem, z):
        # res = self.res_fn(mem, z)
        # gate = self.gate_fn(mem, z)
        # return mem + res*gate
        return z#mem*(1-gate) + res*gate

    def gate_fn(self, mem, z):
        mem_z = torch.cat([mem, z], dim=-1)
        return self.gate(mem_z)
    
    def res_fn(self, mem, z):
        mem_z = torch.cat([mem, z], dim=-1)
        return self.res(mem_z)
    
class MultiIterDEQFixedPointLayer(nn.Module):
    def __init__(self, f, num_iters, **kwargs):
        super().__init__()
        self.f = f
        self.num_iters = num_iters
        self.kwargs = kwargs
        self.grad_type = kwargs.get('grad_type', 'bptt')
    
    def forward(self, x, zi=None):
        if zi is None:
            zi = torch.zeros_like(x)
        for i in range(self.num_iters):
            if self.grad_type == 'bptt':
                zi = self.f(x, zi)
            elif self.grad_type == 'last_step_grad':
                with torch.set_grad_enabled(i == self.num_iters-1):
                    zi = self.f(x, zi)
        return zi

class DEQFixedPointLayer(nn.Module):
    def __init__(self, f, fp_type='anderson', **kwargs):
        super().__init__()
        self.fp_type = fp_type
        if fp_type == 'single':
            self.fp_operator = f
        elif fp_type == 'multi':
            num_iters = kwargs.get('inner_deq_iters', 5)
            self.fp_operator = MultiIterDEQFixedPointLayer(f, num_iters, **kwargs)
        if fp_type == 'broyden':
            solver = broyden
            self.fp_operator = DEQFixedPoint(f, solver, **kwargs)
        elif fp_type == 'anderson':
            solver = anderson
            self.fp_operator = DEQFixedPoint(f, solver, **kwargs)

    def forward(self, x, zi=None, iter=0):
        if self.fp_type == 'single' or self.fp_type == 'multi':
            logs = {"forward_steps": None, "forward_rel_err": None}
            z = self.fp_operator(x, zi)
        elif self.fp_type == 'broyden' or self.fp_type == 'anderson':
            z, logs = self.fp_operator(x, zi, iter)
        return z, logs

class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.grad_type = kwargs['grad_type']
        self.kwargs = kwargs
        
    def forward(self, x, zi=None, iter=0):
        # compute forward pass and re-engage autograd tape
        if zi is None:
            zi = torch.zeros_like(x)
        with torch.no_grad():
            z, self.forward_res, self.forward_steps, self.forward_rel_err = self.solver(lambda z : self.f(x, z), zi, solver_name=f'forward{iter}', **self.kwargs)
        z = self.f(x, z)
        
        # set up Jacobian vector product (without additional forward calls)
        if self.grad_type == 'fp_grad':
            z0 = z.clone().detach().requires_grad_()
            f0 = self.f(x, z0)
        def backward_hook(grad):
            # ipdb.set_trace()
            if self.grad_type == 'last_step_grad':
                return grad
            elif self.grad_type == 'fp_grad':
                g, self.backward_res, self.backward_steps, self.backward_rel_err = self.solver(lambda y : autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                                   grad, solver_name=f'backward{iter}', **self.kwargs)
            return g

        z.register_hook(backward_hook)
        logs = {'forward_steps': self.forward_steps, 'forward_rel_err': self.forward_rel_err}
        return z, logs
    
if __name__ == "__main__":
    # Test the GradNormLayer
    input_size = 3
    grad_norm = GradNormLayer(input_size)
    input = torch.randn(4, input_size, requires_grad=True)
    output = grad_norm(input)
    output.sum().backward()
    print(grad_norm.gradient_moving_avg)
    print(input.grad)