// PyTorch extension header
#include "torch/extension.h"


// C++ function declarations
torch::Tensor naive_softmax(torch::Tensor X);
torch::Tensor better_softmax(torch::Tensor X);


// Bindings for C++ functions
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_softmax", &naive_softmax, "naive softmax kernel");
    m.def("better_softmax", &better_softmax, "better softmax kernel");
}
