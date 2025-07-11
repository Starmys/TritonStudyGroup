# Write a Triton Softmax Kernel

## Quick Start

- Requirements
    - 1 NVIDIA GPU
    - CUDA >= 11.8
    - torch >= 2.0

- Run unit test
    ```bash
    python test.py
    ```

## Homework

- Implement a Triton softmax kernel for arbitrary hidden dimension
  
  Hint: apply mask on the N dimension

- Which things have changed when we set `BLOCK_SIZE_M=1`?

  Hint: what should we do to exchange data across warps?

