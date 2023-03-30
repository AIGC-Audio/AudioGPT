# pragma once

#include <stdint.h>
#include <torch/torch.h>

// _backend.freq_encode_forward(inputs, B, input_dim, degree, output_dim, outputs)
void freq_encode_forward(at::Tensor inputs, const uint32_t B, const uint32_t D, const uint32_t deg, const uint32_t C, at::Tensor outputs);

// _backend.freq_encode_backward(grad, outputs, B, input_dim, degree, output_dim, grad_inputs)
void freq_encode_backward(at::Tensor grad, at::Tensor outputs, const uint32_t B, const uint32_t D, const uint32_t deg, const uint32_t C, at::Tensor grad_inputs);