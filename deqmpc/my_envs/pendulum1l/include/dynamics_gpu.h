
#ifndef DYNAMICS_GPU_H_
#define DYNAMICS_GPU_H_

#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>


std::vector<torch::Tensor> dynamics_gpu(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
std::vector<torch::Tensor> derivatives_gpu(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);

#endif


  