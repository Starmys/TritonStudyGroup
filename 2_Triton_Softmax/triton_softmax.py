import torch
import triton
import triton.language as tl


# Triton kernel for softmax operation
@triton.jit
def softmax_kernel(
    x_ptr,  # Pointer to input tensor
    y_ptr,  # Pointer to output tensor
    batch_size,
    hidden_dim,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for M (batch_size) dimension
    BLOCK_SIZE_N: tl.constexpr,  # Block size for N (hidden_dim) dimension
):
    # Equivalent to blockIdx in CUDA
    pid = tl.program_id(0)

    # Boundary check
    start_m = pid * BLOCK_SIZE_M
    if start_m >= batch_size:
        return

    # Offsets for M and N dimensions
    offs_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    # Mask invalid M offsets when BLOCK_SIZE_M > 1
    mask = offs_m[:, None] < batch_size

    # Load input tensor
    x = tl.load(x_ptr + offs_m[:, None] * hidden_dim + offs_n[None, :], mask=mask)

    # Compute softmax
    m = tl.max(x, axis=1, keep_dims=True)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=1, keep_dims=True)
    y = e / s

    # Store to the output tensor
    tl.store(y_ptr + offs_m[:, None] * hidden_dim + offs_n[None, :], y, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    batch_size, hidden_dim = x.shape
    y = torch.empty_like(x)

    # Program (thread block) size = 4 warps = 128 threads
    num_warps = 4

    # Each program (thread block) processes 4 rows
    BLOCK_SIZE_M = 1
    BLOCK_SIZE_N = hidden_dim
    assert hidden_dim & (hidden_dim - 1) == 0, "Hidden dimension must be a power of 2"

    # Launch the Triton kernel
    grid = (triton.cdiv(batch_size, BLOCK_SIZE_M), )
    softmax_kernel[grid](
        x, y, batch_size, hidden_dim,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=num_warps,
    )

    return y
