# Triton Flash Attention Kernel (Forward & Backward)

## Quick Start

- Requirements
    - 1 NVIDIA GPU
    - CUDA >= 12.1
    - torch == 2.6.0
    - triton == 3.2.0

- Run unit test
    ```bash
    python test.py
    ```

## Homework

- Implement Triton block-sparse Flash-Attention forward and backward kernel with block size of 128

  Hint: Convert the block mask to active block list first

- Implement a Triton Flash-Attention forward kernel that outputs the attention scores after pooling

  Hint: DO NOT output the $N^2$ attention score matrix which may cause OOM

- Calculate the FLOPs of `triton_split_bwd`

  Hint: how many GeMMs are there?

- Why do we split the main loop into 3 parts (but not 2 parts) in the backward kernel?

  Hint: Consider the backward tiling strategy and the causal mask
