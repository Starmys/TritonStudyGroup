#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <float.h>
#include <cuda.h>


#define WARP_SIZE 32          // Number of threads in a warp
#define WARP_MASK 0xFFFFFFFF  // Mask for all threads in a warp: 0xFFFFFFFF = 0b11111111111111111111111111111111
#define MEM_ACCESS_WIDTH 4    // Number of floats accessed in a single memory operation (4 floats = 16 bytes = 128 bits)


template<int VALS_PER_THREAD>  // Each thread (lane) processes VALS_PER_THREAD values
__global__ void better_softmax_kernel(float* x, float* y, int batch_size) {
    // Current warp index in the thread block
    const int warp_idx = threadIdx.x / WARP_SIZE;
    // Current thread lane index within the warp
    const int lane_idx = threadIdx.x % WARP_SIZE;
    // Number of warps in a thread block
    const int num_warps = blockDim.x / WARP_SIZE;

    // Each warp processes one row
    const int row_idx = blockIdx.x * num_warps + warp_idx;
    // Boundary check
    if (row_idx >= batch_size) return;

    // Offset for contiguous memory access in a warp
    const int offset = row_idx * (WARP_SIZE * VALS_PER_THREAD) + lane_idx * MEM_ACCESS_WIDTH;

    // Allocate VALS_PER_THREAD floats in registers
    float tmp_val[VALS_PER_THREAD];

    // Load VALS_PER_THREAD values from global memory into registers
    #pragma unroll
    for (int i = 0; i < VALS_PER_THREAD; i += MEM_ACCESS_WIDTH) {
        // Vectorized memory access using float4
        reinterpret_cast<float4*>(&tmp_val[i])[0] =
            reinterpret_cast<float4*>(&x[offset + i * WARP_SIZE])[0];
    }

    // Find the maximum value in the thread's values
    float max_val = -FLT_MAX;
    #pragma unroll
    for (int i = 0; i < VALS_PER_THREAD; i++) {
        max_val = max(max_val, tmp_val[i]);
    }
    // Reduce the maximum value across all threads in the warp
    #pragma unroll
    for (int laneMask = 1; laneMask < WARP_SIZE; laneMask <<= 1) {
        max_val = max(max_val, __shfl_xor_sync(WARP_MASK, max_val, laneMask));
    }

    // Calculate exponential values and sum of exponentials in the thread's values
    float sum_exp = 0.0f;
    #pragma unroll
    for (int i = 0; i < VALS_PER_THREAD; i++) {
        tmp_val[i] = expf(tmp_val[i] - max_val);
        sum_exp += tmp_val[i];
    }
    // Reduce the sum of exponentials across all threads in the warp
    #pragma unroll
    for (int laneMask = 1; laneMask < WARP_SIZE; laneMask <<= 1) {
        sum_exp += __shfl_xor_sync(WARP_MASK, sum_exp, laneMask);
    }

    // Calculate the softmax values
    #pragma unroll
    for (int i = 0; i < VALS_PER_THREAD; i++) {
        tmp_val[i] /= sum_exp;
    }

    // Write VALS_PER_THREAD values registers to global memory 
    #pragma unroll
    for (int i = 0; i < VALS_PER_THREAD; i += MEM_ACCESS_WIDTH) {
        // Vectorized memory access using float4
        reinterpret_cast<float4*>(&y[offset + i * WARP_SIZE])[0] =
            reinterpret_cast<float4*>(&tmp_val[i])[0];
    }
}


// C++ function to call the better softmax kernel
torch::Tensor better_softmax(torch::Tensor X) {
    cudaSetDevice(X.get_device());

    int batch_size = X.size(0);
    int hidden_dim = X.size(1);
    torch::Tensor Y = torch::empty_like(X, X.options());

    // Number of warps in a thread block
    const int num_warps = 4;
    // Grid size (= number of thread blocks)
    int num_blocks = (batch_size + num_warps - 1) / num_warps;

    const dim3 dimBlock(num_warps * WARP_SIZE);
    const dim3 dimGrid(num_blocks);

    // Launch the kernel with the appropriate vals_per_thread
    if (hidden_dim % WARP_SIZE != 0) {
        throw std::runtime_error("Unsupported hidden dimension size.");
    }
    int vals_per_thread = hidden_dim / WARP_SIZE;
    if (vals_per_thread == 4) {
        better_softmax_kernel<4><<<dimGrid, dimBlock>>>(X.data_ptr<float>(), Y.data_ptr<float>(), batch_size);
    } else if (vals_per_thread == 8) {
        better_softmax_kernel<8><<<dimGrid, dimBlock>>>(X.data_ptr<float>(), Y.data_ptr<float>(), batch_size);
    } else if (vals_per_thread == 16) {
        better_softmax_kernel<16><<<dimGrid, dimBlock>>>(X.data_ptr<float>(), Y.data_ptr<float>(), batch_size);
    } else if (vals_per_thread == 32) {
        better_softmax_kernel<32><<<dimGrid, dimBlock>>>(X.data_ptr<float>(), Y.data_ptr<float>(), batch_size);
    } else if (vals_per_thread == 64) {
        better_softmax_kernel<64><<<dimGrid, dimBlock>>>(X.data_ptr<float>(), Y.data_ptr<float>(), batch_size);
    } else if (vals_per_thread == 128) {
        better_softmax_kernel<128><<<dimGrid, dimBlock>>>(X.data_ptr<float>(), Y.data_ptr<float>(), batch_size);
    } else {
        throw std::runtime_error("Unsupported hidden dimension size.");
    }

    return Y;
}
