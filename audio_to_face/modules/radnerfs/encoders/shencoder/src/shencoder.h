# pragma once

#include <stdint.h>
#include <torch/torch.h>

// inputs: [B, D], float, in [-1, 1]
// outputs: [B, F], float

void sh_encode_forward(at::Tensor inputs, at::Tensor outputs, const uint32_t B, const uint32_t D, const uint32_t C, at::optional<at::Tensor> dy_dx);
void sh_encode_backward(at::Tensor grad, at::Tensor inputs, const uint32_t B, const uint32_t D, const uint32_t C, at::Tensor dy_dx, at::Tensor grad_inputs);