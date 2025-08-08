import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,  # Pointers to input and output matrices
    M, N, K,  # Matrix dimensions
    stride_am, stride_ak,  # Strides for A matrix
    stride_bk, stride_bn,  # Strides for B matrix
    stride_cm, stride_cn,  # Strides for C matrix
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  # Block sizes
):
    # Thread block index
    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    # Create pointers for the first blocks of A and B
    offs_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m % M, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n % N, BLOCK_SIZE_N), BLOCK_SIZE_N)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Iterate to compute a block of the C matrix
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    mid = (K // BLOCK_SIZE_K) * BLOCK_SIZE_K
    for k in range(0, mid, BLOCK_SIZE_K):
        a = tl.load(a_ptrs)  # Load a block of A to shared memory
        b = tl.load(b_ptrs)  # Load a block of B to shared memory
        accumulator = tl.dot(a, b, accumulator)  # Call tensor core
        a_ptrs += BLOCK_SIZE_K * stride_ak  # Move to the next block of A
        b_ptrs += BLOCK_SIZE_K * stride_bk  # Move to the next block of B
    if mid < K:
        mask_k = mid + offs_k < K  # Mask for valid K indices
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.)  # Load a block of A to shared memory
        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.)  # Load a block of B to shared memory
        accumulator = tl.dot(a, b, accumulator)  # Call tensor core
    c = accumulator.to(c_ptr.type.element_ty)  # Convert to the output dtype

    # Write back the block of the output matrix C
    offs_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Inner dimensions must match for matrix multiplication"
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64
    grid = (triton.cdiv(N, BLOCK_SIZE_N), triton.cdiv(M, BLOCK_SIZE_M), )
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        num_stages=3, num_warps=8,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return c
