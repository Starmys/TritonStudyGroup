# Write a Simple CUDA Kernel: Softmax

## Problem Definition

- Shape: batch size $M \ge 4096$, hidden dimension $N = 4096$

- Mathematical

  $$X \in \mathbb{R}^{M \times N}$$

  $$Y_{ij} = \mathrm{softmax}(X_i)_j$$

  $$= \frac{\exp(X_{ij})}{\sum_{j=1}^{N}\exp(X_{ij})}$$

- For numerical stability

  $$X \in \mathbb{R}^{M \times N}_{\mathrm{float32}}$$

  $$M_i \gets \max_{j=1}^{N} X_{ij}$$

  $$Z_{ij} \gets \exp(X_{ij}-M_i)$$

  $$S_i \gets \sum_{j=1}^{N}Z_{ij}$$

  $$Y_{ij} \gets \frac{Z_{ij}}{S_i}$$


## Custom Softmax Kernels

- `naive_softmax()`
    - 1 thread per row, small grid, limited registers
    - Repeated global memory access to calculate $M_i$ and $S_i$
    - No collaboration between threads
- `better_softmax()`
    - 1 warp per row, large grid, sufficient registers
    - Vectorized one-time global memory access
    - Apply shuffle-sync for in-warp collaboration


## Quick Start

- Requirements
    - 1 NVIDIA GPU
    - CUDA >= 11.8
    - torch >= 2.0

- Installation
    ```bash
    cd 1_CUDA_Softmax
    # It takes ~ 1 minute to compile kernels
    pip install -e .
    ```

- Run unit test
    ```bash
    python test.py
    ```

## Homework

- Calculate the theoretical minimum latency for input size [8765, 4096] on single A100 GPU
  
  Hint: check [NVIDIA A100 Specifications](https://www.nvidia.com/en-us/data-center/a100/) for global memory bandwidth

- Can we load the entire row into L1 cache or shared memory in naïve softmax?

  Hint: check [NVIDIA Ampere Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) for L1 cache size

- How much acceleration does warp-level contiguous global memory access bring? Why?

  Hint: modify `better_softmax_kernel()` and profile the latancy with non-contiguous memory access

- How does the best softmax implementation changes for inputs of different shapes?

  Hint: try different input shapes in `test.py` first
