#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <float.h>
#include <cuda.h>


#define WARP_SIZE 32
#define WARP_MASK 0xFFFFFFFF
#define MEM_ACCESS_WIDTH 4


template<int VALS_PER_THREAD>
__global__ void better_softmax_kernel(float* x, float* y, int batch_size) {
    const int warp_idx = threadIdx.x / WARP_SIZE;
    const int lane_idx = threadIdx.x % WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    const int row_idx = blockIdx.x * num_warps + warp_idx;
    if (row_idx >= batch_size) return;

    const int offset = (row_idx * WARP_SIZE + lane_idx) * VALS_PER_THREAD;

    float tmp_val[VALS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < VALS_PER_THREAD; i += MEM_ACCESS_WIDTH) {
        reinterpret_cast<float4*>(&tmp_val[i])[0] =
            reinterpret_cast<float4*>(&x[offset + i])[0];
    }

    float max_val = -FLT_MAX;
    #pragma unroll
    for (int i = 0; i < VALS_PER_THREAD; i++) {
        max_val = max(max_val, tmp_val[i]);
    }
    #pragma unroll
    for (int laneMask = 1; laneMask < WARP_SIZE; laneMask <<= 1) {
        max_val = max(max_val, __shfl_xor_sync(WARP_MASK, max_val, laneMask));
    }

    float sum_exp = 0.0f;
    #pragma unroll
    for (int i = 0; i < VALS_PER_THREAD; i++) {
        tmp_val[i] = expf(tmp_val[i] - max_val);
        sum_exp += tmp_val[i];
    }
    #pragma unroll
    for (int laneMask = 1; laneMask < WARP_SIZE; laneMask <<= 1) {
        sum_exp += __shfl_xor_sync(WARP_MASK, sum_exp, laneMask);
    }

    #pragma unroll
    for (int i = 0; i < VALS_PER_THREAD; i++) {
        tmp_val[i] /= sum_exp;
    }

    #pragma unroll
    for (int i = 0; i < VALS_PER_THREAD; i += MEM_ACCESS_WIDTH) {
        reinterpret_cast<float4*>(&y[offset + i])[0] =
            reinterpret_cast<float4*>(&tmp_val[i])[0];
    }
}


torch::Tensor better_softmax(torch::Tensor X) {
    cudaSetDevice(X.get_device());

    int batch_size = X.size(0);
    int hidden_dim = X.size(1);
    torch::Tensor Y = torch::empty_like(X, X.options());

    const int num_warps = 4;
    int num_blocks = (batch_size + num_warps - 1) / num_warps;

    const dim3 dimBlock(num_warps * WARP_SIZE);
    const dim3 dimGrid(num_blocks);

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
