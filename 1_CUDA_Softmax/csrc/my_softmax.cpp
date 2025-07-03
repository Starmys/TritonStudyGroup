#include <vector>
#include "torch/extension.h"


at::Tensor naive_softmax(torch::Tensor X);
// at::Tensor better_softmax(torch::Tensor X);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_softmax", &naive_softmax, "naive softmax kernel");
    // m.def("better_softmax", &better_softmax, "better softmax kernel");
}
