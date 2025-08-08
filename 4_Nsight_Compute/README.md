# Write a Triton GeMM Kernel

## Quick Start

- Requirements
    - 1 NVIDIA GPU
    - CUDA >= 12.1
    - torch == 2.5.0
    - triton == 3.1.0

- Get best config for previous triton GeMM kernel
    ```bash
    cd ../3_Triton_GeMM
    TRITON_PRINT_AUTOTUNING=1 python test.py
    ```

- Run NCU profile
    ```bash
    bash run_profile.sh
    ```

- Run unit test
    ```bash
    python test.py
    ```

## Homework

- Play with NCU
