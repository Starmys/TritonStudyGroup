# Write a Triton GeMM Kernel

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

- Implement a Triton linear kernel ($C=AB^T$)
  
  Hint: set correct stride input to transpose B

- How to estimate `num_stages` under a given problem shape?

  Hint: how much faster does L2 cache compare to global memory access?
