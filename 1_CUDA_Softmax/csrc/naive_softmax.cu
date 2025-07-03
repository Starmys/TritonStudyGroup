#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <float.h>
#include <cuda.h>


// CUDA kernel for naive softmax implementation
__global__ void naive_softmax_kernel(float* x, float* y, int batch_size, int hidden_dim) {
    // Each thread processes one row of the input matrix x
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Boundary check
    if (row_idx >= batch_size) return;

    // Calculate the maximum value in the row
    float max_val = -FLT_MAX;
    for (int i = 0; i < hidden_dim; i++) {
        float tmp_val = x[row_idx * hidden_dim + i];  // Read from global memory
        max_val = max(max_val, tmp_val);
    }

    // Calculate the sum of exponentials
    float sum_exp = 0.0f;
    for (int i = 0; i < hidden_dim; i++) {
        float tmp_val = x[row_idx * hidden_dim + i];  // Read from global memory
        sum_exp += expf(tmp_val - max_val);
    }

    // Write the softmax output
    for (int i = 0; i < hidden_dim; i++) {
        float tmp_val = x[row_idx * hidden_dim + i];  // Read from global memory
        y[row_idx * hidden_dim + i] = expf(tmp_val - max_val) / sum_exp;  // Write to global memory
    }
}


// C++ function to call the naive softmax kernel
torch::Tensor naive_softmax(torch::Tensor X) {
    cudaSetDevice(X.get_device());

    int batch_size = X.size(0);
    int hidden_dim = X.size(1);
    torch::Tensor Y = torch::empty_like(X, X.options());

    // Thread block size
    const int num_threads = 128;
    // Grid size (= number of thread blocks)
    int num_blocks = (batch_size + num_threads - 1) / num_threads;

    // Launch the kernel
    const dim3 dimBlock(num_threads);
    const dim3 dimGrid(num_blocks);
    naive_softmax_kernel<<<dimGrid, dimBlock>>>(
        X.data_ptr<float>(),  // Pointer to input data
        Y.data_ptr<float>(),  // Pointer to output data
        batch_size, hidden_dim
    );

    return Y;
}
