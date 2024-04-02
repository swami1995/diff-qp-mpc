#include <torch/extension.h>
#include "dynamics_gpu.h"
#include "dynamics_cpu.h"

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

/* Interface for cuda and c++ robot dynamics

*/

// Device-agnostic torch wrapper
std::vector<torch::Tensor> dynamics(const torch::Tensor q_in, const torch::Tensor qdot_in, const torch::Tensor tau_in, const torch::Tensor h_in)
{
    CHECK_CONTIGUOUS(q_in);
    CHECK_CONTIGUOUS(qdot_in);
    CHECK_CONTIGUOUS(tau_in);
    CHECK_CONTIGUOUS(h_in);

    if (q_in.device().type() == torch::DeviceType::CPU)
    {
        return dynamics_cpu(q_in, qdot_in, tau_in, h_in);
    }
    else if (q_in.device().type() == torch::DeviceType::CUDA)
    {
        return dynamics_gpu(q_in, qdot_in, tau_in, h_in);
    }

    return {};
}

std::vector<torch::Tensor> derivatives(const torch::Tensor q_in, const torch::Tensor qdot_in, const torch::Tensor tau_in, const torch::Tensor h_in)
{
    CHECK_CONTIGUOUS(q_in);
    CHECK_CONTIGUOUS(qdot_in);
    CHECK_CONTIGUOUS(tau_in);
    CHECK_CONTIGUOUS(h_in);

    if (q_in.device().type() == torch::DeviceType::CPU)
    {
        return derivatives_cpu(q_in, qdot_in, tau_in, h_in);
    }
    else if (q_in.device().type() == torch::DeviceType::CUDA)
    {
        return derivatives_gpu(q_in, qdot_in, tau_in, h_in);
    }

    return {};
}

// Define the Python module and bindings using Pybind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("dynamics", &dynamics, "Forward dynamics function");
    m.def("derivatives", &derivatives, "Forward derivatives function");
}