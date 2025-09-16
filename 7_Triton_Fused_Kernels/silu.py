import torch
import triton
import triton.language as tl
import pandas as pd


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 4096}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 2, 'BLOCK_SIZE_N': 2048}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 1024}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 512}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256}, num_warps=8, num_stages=1),
    ],
    key=['N'],
)
@triton.jit
def _fused_silu_kernel(
    x_ptr, y_ptr,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    M, N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=0)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = offs_m < M
    offs_m %= M
    x1_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x3_ptrs = x1_ptrs + N * stride_xn
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    x1 = tl.load(x1_ptrs).to(tl.float32)
    x3 = tl.load(x3_ptrs).to(tl.float32)
    y = x1 * (1 / (1 + tl.exp(-x1))) * x3
    tl.store(y_ptrs, y.to(y_ptr.type.element_ty), mask=mask[:, None])


def fused_silu(inputs: torch.Tensor):
    num_tokens, hidden_dim = inputs.shape
    hidden_dim //= 2
    outputs = torch.empty((num_tokens, hidden_dim), device=inputs.device, dtype=inputs.dtype)
    _fused_silu_kernel[lambda META: (
        triton.cdiv(hidden_dim, META['BLOCK_SIZE_N']),
        triton.cdiv(num_tokens, META['BLOCK_SIZE_M']),
    )](
        inputs, outputs,
        inputs.stride(0), inputs.stride(1),
        outputs.stride(0), outputs.stride(1),
        num_tokens, hidden_dim,
    )
    return outputs


def naive_silu(inputs: torch.Tensor):
    hidden_dim = inputs.shape[-1] // 2
    x1 = inputs[:, :hidden_dim]
    x3 = inputs[:, hidden_dim:]
    return x1 * (1 / (1 + torch.exp(-x1))) * x3


@torch.compile
def compiled_silu(inputs: torch.Tensor):
    hidden_dim = inputs.shape[-1] // 2
    x1 = inputs[:, :hidden_dim]
    x3 = inputs[:, hidden_dim:]
    return x1 * (1 / (1 + torch.exp(-x1))) * x3


def test_silu(num_tokens: int, hidden_dim: int):
    torch.manual_seed(4321)
    inputs = torch.randn((num_tokens, hidden_dim * 2), dtype=torch.float32, device='cuda')

    o_triton = fused_silu(inputs)
    o_naive = naive_silu(inputs)
    o_compiled = compiled_silu(inputs)

    atol = 1e-2
    rtol = 1e-3
    torch.testing.assert_close(o_triton, o_naive, atol=atol, rtol=rtol)
    torch.testing.assert_close(o_compiled, o_naive, atol=atol, rtol=rtol)

    print(f'Problem size: ({num_tokens}, {hidden_dim})')
    print(pd.DataFrame({'latency': {
        'naive_silu': triton.testing.do_bench(lambda: naive_silu(inputs)),
        'compiled_silu': triton.testing.do_bench(lambda: compiled_silu(inputs)),
        'fused_silu': triton.testing.do_bench(lambda: fused_silu(inputs)),
    }}))


if __name__ == "__main__":
    test_silu(num_tokens=4096, hidden_dim=4096)
