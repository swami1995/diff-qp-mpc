import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

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
    
    
if __name__ == "__main__":
    # Test the GradNormLayer
    input_size = 3
    grad_norm = GradNormLayer(input_size)
    input = torch.randn(4, input_size, requires_grad=True)
    output = grad_norm(input)
    output.sum().backward()
    print(grad_norm.gradient_moving_avg)
    print(input.grad)