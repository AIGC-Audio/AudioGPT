#ifndef _HASH_ENCODE_H
#define _HASH_ENCODE_H

#include <stdint.h>
#include <torch/torch.h>

// inputs: [B, D], float, in [0, 1]
// embeddings: [sO, C], float
// offsets: [L + 1], uint32_t
// outputs: [B, L * C], float
// H: base resolution
void grid_encode_forward(const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, at::optional<at::Tensor> dy_dx, const uint32_t gridtype, const bool align_corners, const uint32_t interp);
void grid_encode_backward(const at::Tensor grad, const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor grad_embeddings, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const at::optional<at::Tensor> dy_dx, at::optional<at::Tensor> grad_inputs, const uint32_t gridtype, const bool align_corners, const uint32_t interp);

void grad_total_variation(const at::Tensor inputs, const at::Tensor embeddings, at::Tensor grad, const at::Tensor offsets, const float weight, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const uint32_t gridtype, const bool align_corners);

#endif