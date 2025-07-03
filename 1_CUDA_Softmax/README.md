# Write a Simple CUDA Kernel: Softmax

## Problem Defination

- Shape: $M$ is batch size, $N$ is hidden dimension
$$4096 <= M < 16384, N = 4096$$

- Mathematical

  $$X \in \mathbb{R}^{M \times N}$$

  $$Y_{ij} = \mathrm{softmax}(X_i)_j$$

  $$= \frac{\exp(X_{ij})}{\sum_{j=1}^{N}\exp(X_{ij})}$$

- For numerical stability

  $$X \in \mathbb{R}^{M \times N}_{\mathrm{float32}}$$

  $$M_i \gets \max_{j=0}^{N} X_{ij}$$

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
