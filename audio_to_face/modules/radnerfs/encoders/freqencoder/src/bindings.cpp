#include <torch/extension.h>

#include "freqencoder.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("freq_encode_forward", &freq_encode_forward, "freq encode forward (CUDA)");
    m.def("freq_encode_backward", &freq_encode_backward, "freq encode backward (CUDA)");
}