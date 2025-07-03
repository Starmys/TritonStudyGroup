#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <float.h>
#include <cuda.h>


__global__ void naive_softmax_kernel(
    float* x,
    float* y,
    int M,
    int N
) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= M) return;

    float max_val = -FLT_MAX;
    for (int i = 0; i < N; i++) {
        float tmp_val = x[row_idx * N + i];
        max_val = tmp_val > max_val ? tmp_val : max_val;
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < N; i++) {
        float tmp_val = x[row_idx * N + i];
        sum_exp += expf(tmp_val - max_val);
    }

    for (int i = 0; i < N; i++) {
        float tmp_val = x[row_idx * N + i];
        y[row_idx * N + i] = expf(tmp_val - max_val) / sum_exp;
    }
}


at::Tensor naive_softmax(torch::Tensor X) {
    cudaSetDevice(X.get_device());

    int batch_size = X.size(0);
    int hidden_dim = X.size(1);
    torch::Tensor Y = torch::empty_like(X, X.options());

    const int num_threads = 128;
    int num_blocks = (batch_size + num_threads - 1) / num_threads;

    const dim3 dimBlock(num_threads);
    const dim3 dimGrid(num_blocks);
    naive_softmax_kernel<<<dimGrid, dimBlock>>>(
        X.data_ptr<float>(),
        Y.data_ptr<float>(),
        batch_size, hidden_dim
    );

    return Y;
}
