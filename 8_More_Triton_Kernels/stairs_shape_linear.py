import torch
import torch.nn as nn
import numpy as np
import math


import triton
import triton.language as tl

from typing import List


def convert_heads_to_dims(
    dim: int,
    heads: List[int] = [8, 8, 16],
    cumsum: bool = False
) -> np.array:
    """Map matryoshka heads to the corresponding dimensions (inputs / outputs)."""
    assert dim % sum(heads) == 0, f"{dim} should be divisible by the sum of {heads}."
    head_dim = dim // sum(heads)
    if cumsum:
        heads = np.cumsum(heads)
    return np.array(heads) * head_dim


@triton.jit
def _triton_stairs_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    block_count, num_blocks,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_block,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    SPARSE_TENSOR: tl.constexpr, INVERSED: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    if SPARSE_TENSOR == 'C':
        row_idx = (pid_n * BLOCK_SIZE_N) // BLOCK_SIZE
        col_start = tl.load(block_count + row_idx)
        col_end = tl.load(block_count + row_idx + 1)
        if INVERSED:
            if start_m < (num_blocks + col_start - col_end) * BLOCK_SIZE:
                return
        else:
            if start_m >= (col_end - col_start) * BLOCK_SIZE:
                return

    if SPARSE_TENSOR == 'A':
        offs_am = tl.arange(0, BLOCK_SIZE_M)
    else:
        offs_am = (start_m + tl.arange(0, BLOCK_SIZE_M)) % M
    if SPARSE_TENSOR == 'B':
        offs_bn = tl.arange(0, BLOCK_SIZE_N)
    else:
        offs_bn = (start_n + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if SPARSE_TENSOR == 'A':
        row_idx = (pid_m * BLOCK_SIZE_M) // BLOCK_SIZE
        row_off = (pid_m * BLOCK_SIZE_M) % BLOCK_SIZE
        col_start = tl.load(block_count + row_idx)
        col_end = tl.load(block_count + row_idx + 1)
        if INVERSED:
            b_ptrs += ((num_blocks + col_start - col_end) * BLOCK_SIZE) * stride_bk
        a_ptrs += col_start * stride_block + col_off * stride_am
        lo, hi = 0, (col_end - col_start) * (BLOCK_SIZE // BLOCK_SIZE_K)
        if INVERSED:
            hi += tl.cdiv(K, BLOCK_SIZE_K) - num_blocks * (BLOCK_SIZE // BLOCK_SIZE_K)
    elif SPARSE_TENSOR == 'B':
        row_idx = (pid_n * BLOCK_SIZE_N) // BLOCK_SIZE
        row_off = (pid_n * BLOCK_SIZE_N) % BLOCK_SIZE
        col_start = tl.load(block_count + row_idx)
        col_end = tl.load(block_count + row_idx + 1)
        if INVERSED:
            a_ptrs += ((num_blocks + col_start - col_end) * BLOCK_SIZE) * stride_ak
        b_ptrs += col_start * stride_block + row_off * stride_bn
        lo, hi = 0, (col_end - col_start) * (BLOCK_SIZE // BLOCK_SIZE_K)
        if INVERSED:
            hi += tl.cdiv(K, BLOCK_SIZE_K) - num_blocks * (BLOCK_SIZE // BLOCK_SIZE_K)
    else:
        lo, hi = 0, K // BLOCK_SIZE_K
    for k in range(lo, hi):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, accumulator)
        if SPARSE_TENSOR == 'A' and (k + 1) % (BLOCK_SIZE // BLOCK_SIZE_K) == 0:
            a_ptrs += stride_block + (BLOCK_SIZE_K - BLOCK_SIZE) * stride_ak
        else:
            a_ptrs += BLOCK_SIZE_K * stride_ak
        if SPARSE_TENSOR == 'B' and (k + 1) % (BLOCK_SIZE // BLOCK_SIZE_K) == 0:
            b_ptrs += stride_block + (BLOCK_SIZE_K - BLOCK_SIZE) * stride_bk
        else:
            b_ptrs += BLOCK_SIZE_K * stride_bk
    if SPARSE_TENSOR == 'C':
        for k in range(hi, tl.cdiv(K, BLOCK_SIZE_K)):
            mask = offs_k < (K % BLOCK_SIZE_K)
            a = tl.load(a_ptrs, mask=mask[None, :], other=0.)
            b = tl.load(b_ptrs, mask=mask[:, None], other=0.)
            accumulator = tl.dot(a, b, accumulator)
    c = accumulator.to(a_ptr.dtype.element_ty)

    if SPARSE_TENSOR == 'C':
        row_off = (pid_n * BLOCK_SIZE_N) % BLOCK_SIZE
        col_idx = (pid_m * BLOCK_SIZE_M) // BLOCK_SIZE
        if INVERSED:
            col_idx -= num_blocks + col_start - col_end
        col_off = (pid_m * BLOCK_SIZE_M) % BLOCK_SIZE
        offs_cm = col_off + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = row_off + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + (col_start + col_idx) * stride_block + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        tl.store(c_ptrs, c)
    else:
        offs_cm = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = start_n + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _triton_reorder_kernel(
    x_ptr, y_ptr, map_idx,
    stride_xz, stride_xr, stride_xc,
    stride_yz, stride_yr, stride_yc,
    SPLIT_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    src_idx = tl.program_id(1)
    tgt_idx = tl.load(map_idx + src_idx)
    split_idx = tl.program_id(0)
    offs_r = split_idx * SPLIT_SIZE + tl.arange(0, SPLIT_SIZE)
    offs_c = tl.arange(0, BLOCK_SIZE)
    x_ptrs = x_ptr + src_idx * stride_xz + offs_r[:, None] * stride_xr + offs_c[None, :] * stride_xc
    y_ptrs = y_ptr + tgt_idx * stride_yz + offs_r[:, None] * stride_yr + offs_c[None, :] * stride_yc
    tl.store(y_ptrs, tl.load(x_ptrs))


class TritonStairsLinearFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        inputs,
        weight,
        row_cnt,
        col_cnt,
        col_map,
        block_size,
        max_output_blocks,
    ):
        M, K = inputs.shape
        N = max_output_blocks * block_size
        # assert K == (col_cnt.shape[0] - 1) * block_size
        num_rows = row_cnt.shape[0] - 1
        num_cols = col_cnt.shape[0] - 1
        block_M, block_N, block_K = 128, 128, 32
        assert block_size % block_M == 0
        assert block_size % block_N == 0
        assert block_size % block_K == 0
        outputs = torch.empty((M, N), device=inputs.device, dtype=inputs.dtype)
        grid = (triton.cdiv(M, block_M) * triton.cdiv(N, block_N), 1, 1)
        _triton_stairs_matmul_kernel[grid](
            inputs, weight, outputs,
            M, N, K, row_cnt, num_cols,
            inputs.stride(0), inputs.stride(1),
            weight.stride(2), weight.stride(1),
            outputs.stride(0), outputs.stride(1),
            weight.stride(0),
            BLOCK_SIZE_M=block_M, BLOCK_SIZE_N=block_N, BLOCK_SIZE_K=block_K,
            GROUP_SIZE_M=8,
            BLOCK_SIZE=block_size,
            SPARSE_TENSOR='B', INVERSED=False,
            num_warps=4, num_stages=4,
        )
        ctx.save_for_backward(inputs, weight, row_cnt, col_cnt, col_map)
        ctx.block_size = block_size
        ctx.M = M
        ctx.N = N
        ctx.K = K
        ctx.num_rows = num_rows
        ctx.num_cols = num_cols
        ctx.max_output_blocks = max_output_blocks
        return outputs

    @staticmethod
    def backward(ctx, grad, *args):
        inputs, weight, row_cnt, col_cnt, col_map = ctx.saved_tensors
        grad_i, grad_w = None, None
        if ctx.needs_input_grad[0]:
            block_M, block_N, block_K = 128, 128, 32
            assert ctx.block_size % block_M == 0
            assert ctx.block_size % block_N == 0
            assert ctx.block_size % block_K == 0
            grad_i = torch.empty_like(inputs)
            # weight = weight_T[col_map]
            weight_T = torch.empty_like(weight)
            split_size = 8
            grid = (ctx.block_size // split_size, row_cnt[ctx.max_output_blocks].item(), 1)
            _triton_reorder_kernel[grid](
                weight, weight_T, col_map,
                weight.stride(0), weight.stride(1), weight.stride(2),
                weight_T.stride(0), weight_T.stride(1), weight_T.stride(2),
                SPLIT_SIZE=split_size, BLOCK_SIZE=ctx.block_size,
                num_warps=4, num_stages=1,
            )
            grid = (triton.cdiv(ctx.M, block_M) * triton.cdiv(ctx.K, block_N), 1, 1)
            _triton_stairs_matmul_kernel[grid](
                grad, weight_T, grad_i,
                ctx.M, ctx.K, ctx.N, col_cnt, ctx.num_rows,
                grad.stride(0), grad.stride(1),
                weight_T.stride(1), weight_T.stride(2),
                grad_i.stride(0), grad_i.stride(1),
                weight_T.stride(0),
                BLOCK_SIZE_M=block_M, BLOCK_SIZE_N=block_N, BLOCK_SIZE_K=block_K,
                GROUP_SIZE_M=8,
                BLOCK_SIZE=ctx.block_size,
                SPARSE_TENSOR='B', INVERSED=True,
                num_warps=4, num_stages=4,
            )
        if ctx.needs_input_grad[1]:
            block_M, block_N, block_K = 128, 128, 32
            assert ctx.block_size % block_M == 0
            assert ctx.block_size % block_N == 0
            assert ctx.block_size % block_K == 0
            grad_w = torch.zeros_like(weight)
            grid = (triton.cdiv(ctx.K, block_M) * triton.cdiv(ctx.N, block_N), 1, 1)
            _triton_stairs_matmul_kernel[grid](
                inputs, grad, grad_w,
                ctx.K, ctx.N, ctx.M, row_cnt, ctx.num_cols,
                inputs.stride(1), inputs.stride(0),
                grad.stride(0), grad.stride(1),
                grad_w.stride(2), grad_w.stride(1),
                grad_w.stride(0),
                BLOCK_SIZE_M=block_M, BLOCK_SIZE_N=block_N, BLOCK_SIZE_K=block_K,
                GROUP_SIZE_M=8,
                BLOCK_SIZE=ctx.block_size,
                SPARSE_TENSOR='C', INVERSED=False,
                num_warps=4, num_stages=4,
            )
        return grad_i, grad_w, None, None, None, None, None


class TritonMatryoshkaLinear(nn.Module):

    def __init__(
        self, 
        dim: int = 4096,
        hidden_dim: int = None,
        heads: List[int] = [8, 8, 16],
        block_size: int = 128,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or dim
        self.heads = heads
        self.num_heads = sum(self.heads)
        self.num_chains = self.num_heads // self.heads[0]
        assert self.dim % self.num_chains == 0
        assert self.hidden_dim % self.num_chains == 0
        assert block_size % 128 == 0

        self.block_size = block_size
        self.input_dims = convert_heads_to_dims(
            self.dim, heads, cumsum=True
        )
        self.output_dims = convert_heads_to_dims(
            self.hidden_dim, heads
        )
        self._input_blocks = self.input_dims // block_size
        self._output_blocks = np.cumsum(self.output_dims) // block_size

        self.num_blocks = np.sum(self.input_dims * self.output_dims) // (block_size * block_size)
        self.weight = nn.Parameter(torch.empty((self.num_blocks, block_size, block_size)))
        if bias:
            self.bias = nn.Parameter(torch.empty((self.hidden_dim, )))
        else:
            self.bias = None
        self.row_cnt, self.col_cnt, self.col_map = None, None, None

    def init_weights(self, init_std=0.02):
        self.weight.data.normal_(mean=0.0, std=init_std)

    def pre_build_block_index(self):
        if self.row_cnt is None or self.col_cnt is None or self.col_map is None:
            self.row_cnt, self.col_cnt, self.col_map = self._build_block_index()

    def _build_block_index(self):
        num_heads = self.num_chains
        heads = [i // self.heads[0] for i in self.heads]
        num_ci_per_head = self.dim // num_heads
        num_co_per_head = self.hidden_dim // num_heads
        num_cols_per_head = num_ci_per_head // self.block_size
        num_rows_per_head = num_co_per_head // self.block_size
        num_rows = num_rows_per_head * num_heads
        num_cols = num_cols_per_head * num_heads
        block_index = np.zeros((num_rows, num_cols), dtype=int) - 1
        head_cnt = 0
        block_cnt = 0
        row_cnt = [0, ]
        for h in heads:
            for i in range(head_cnt * num_rows_per_head, (head_cnt + h) * num_rows_per_head):
                for j in range(0, (head_cnt + h) * num_cols_per_head):
                    block_index[i, j] = block_cnt
                    block_cnt += 1
                row_cnt.append(block_cnt)
            head_cnt += h
        block_cnt = 0
        col_cnt = [0, ]
        col_map = []
        for j in range(num_cols):
            col_cnt.append
            for i in range(num_rows):
                if block_index[i, j] >= 0:
                    col_map.append(block_index[i, j])
                    block_cnt += 1
            col_cnt.append(block_cnt)
        row_cnt = torch.tensor(row_cnt, dtype=torch.int32, device=self.weight.device)
        col_cnt = torch.tensor(col_cnt, dtype=torch.int32, device=self.weight.device)
        col_map = torch.tensor(col_map, dtype=torch.int64, device=self.weight.device)
        arange = torch.arange(block_cnt, dtype=torch.int32, device=self.weight.device)
        col_map = torch.scatter(input=arange, dim=0, index=col_map, src=arange)
        return row_cnt, col_cnt, col_map

    def forward(self, x: torch.Tensor, head: int = None) -> torch.Tensor:
        self.pre_build_block_index()
        input_shape = x.shape
        x = x.view(-1, input_shape[-1])

        if head is None:
            head = len(self.heads) - 1
        else:
            assert (
                0 <= head < len(self.heads)
            ), f"The head id should be in [0, {len(self.heads)})."
        outputs = TritonStairsLinearFunc.apply(
            x, self.weight,
            self.row_cnt, self.col_cnt, self.col_map,
            self.block_size, self._output_blocks[head].item(),
        )
        outputs = outputs.view(*input_shape[:-1], -1)
        return outputs

    def __repr__(self) -> str:
        inner_string = '\n'.join(
            ['  ({}): Linear({}, {})'.format(i, idim, odim) for i, (idim, odim) in enumerate(zip(self.input_dims, self.output_dims))]
        )
        format_string = self.__class__.__name__ + '([\n' + inner_string + '\n])'
        return format_string

    def to_sparse_weight(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == self.hidden_dim
        assert x.shape[1] == self.dim
        y = []
        for i in range(self.row_cnt.shape[0] - 1):
            start_i = i * self.block_size
            block_cnt = self.row_cnt[i].item()
            num_cols = self.row_cnt[i + 1].item() - block_cnt
            for j in range(num_cols):
                start_j = j * self.block_size
                y.append(x[start_i:start_i+self.block_size, start_j:start_j+self.block_size])
        return torch.stack(y)


class NaiveMatryoshkaLinear(nn.Module):

    def __init__(
        self, 
        dim: int = 4096,
        hidden_dim: int = None,
        heads: List[int] = [8, 8, 16],
        bias=False
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or dim
        self.heads = heads

        self.input_dims = convert_heads_to_dims(
            self.dim, heads, cumsum=True
        )
        self.output_dims = convert_heads_to_dims(
            self.hidden_dim, heads
        )

        self.mlps = nn.ModuleList([
            nn.Linear(indim, outdim, bias=bias)
            for indim, outdim in zip(self.input_dims, self.output_dims)
        ])

    def get_dense_weight(self):
        dense_weight = torch.zeros(
            (self.hidden_dim, self.dim),
            dtype=self.mlps[0].weight.dtype,
            device=self.mlps[0].weight.device,
        )
        dense_mask = torch.zeros_like(dense_weight, dtype=torch.bool)
        c_out_sum = 0
        for layer in self.mlps:
            c_out, c_in = layer.weight.shape
            dense_weight[c_out_sum:c_out_sum+c_out, 0:c_in] = layer.weight
            dense_mask[c_out_sum:c_out_sum+c_out, 0:c_in] = True
            c_out_sum += c_out
        return dense_weight, dense_mask

    def get_sparse_weight(self, block_size: int = 128):
        block_list = []
        num_blocks = [0, ]
        for layer in self.mlps:
            c_out, c_in = layer.weight.shape
            assert c_out % block_size == 0
            assert c_in % block_size == 0
            for start_n in range(0, c_out, block_size):
                for start_k in range(0, c_in, block_size):
                    block_list.append(layer.weight[start_n:start_n+block_size, start_k:start_k+block_size])
                num_blocks.append(c_in // block_size)
        sparse_weight = torch.stack(block_list)
        num_blocks = torch.tensor(num_blocks, dtype=torch.int32, device=sparse_weight.device)
        cumsum_blocks = torch.cumsum(num_blocks, dim=0, dtype=torch.int32)
        return sparse_weight, cumsum_blocks

    def forward(self, x: torch.Tensor, head: int = None) -> torch.Tensor:
        if head is None:
            outputs = [self.mlps[i](x[..., :indim]) for i, indim in enumerate(self.input_dims)]
        else:
            assert (
                0 <= head < len(self.input_dims)
            ), f"The head id should be in [0, {len(self.input_dims)})."
            # Can be optimized
            outputs = [self.mlps[i](x[..., :indim]) for i, indim in enumerate(self.input_dims[:head + 1])]
        return torch.cat(outputs, dim=-1)


def profile(func, inputs, num_warmups=100, num_iters=100):
    torch.cuda.synchronize()
    for _ in range(num_warmups):
        func(*inputs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        func(*inputs)
    end.record()
    torch.cuda.synchronize()
    latency = start.elapsed_time(end) / num_iters
    return latency


def test_masked_linear(
    num_inputs: int = 4096,
    dim: int = 4096,
    hidden_dim: int = 4096,
    heads: int = [8, 8, 16],
    max_head: int = None,
    block_size: int = 128,
    dtype: int = torch.float16,
    device: int = 'cuda',
):
    print(f"num_inputs = {num_inputs}, in_features = {dim}, out_features = {hidden_dim}")
    print(f"heads = {heads}, max_head = {max_head}")
    max_out_features = hidden_dim if max_head is None else hidden_dim * np.cumsum(heads)[max_head] // np.sum(heads)
    inputs = torch.randn((num_inputs, dim), dtype=dtype, device=device, requires_grad=True)
    grad = torch.randn((num_inputs, max_out_features), dtype=dtype, device=device, requires_grad=False) * 1e-1

    naive_linear = NaiveMatryoshkaLinear(dim, hidden_dim, heads).to(dtype).to(device)
    # naive_linear.load_state_dict({k: torch.randn_like(v) for k, v in naive_linear.named_parameters()})
    dense_weight, dense_mask = naive_linear.get_dense_weight()
    sparse_weight, cumsum_blocks = naive_linear.get_sparse_weight()
    print(f"sparsity = {1 - cumsum_blocks[-1].item() / (dense_weight.numel() / (block_size * block_size))}")

    torch_linear = nn.Linear(in_features=dim, out_features=max_out_features, bias=False, dtype=dtype, device=device)
    torch_linear.load_state_dict({'weight': dense_weight[:max_out_features]})

    triton_linear = TritonMatryoshkaLinear(dim, hidden_dim, heads, block_size=block_size).to(dtype).to(device)
    triton_linear.load_state_dict({'weight': sparse_weight})

    def call_linear(inputs, linear, grad=None, backward=False):
        if hasattr(linear, 'heads'):
            outputs = linear(inputs, max_head)
        else:
            outputs = linear(inputs)
        if backward:
            inputs.grad = None
            linear.zero_grad()
            outputs.backward(grad)
            grad_i = inputs.grad.clone() if inputs.requires_grad else None
            return outputs, grad_i
        return outputs

    torch_out, torch_grad_i = call_linear(inputs, torch_linear, grad=grad, backward=True)
    naive_out, naive_grad_i = call_linear(inputs, naive_linear, grad=grad, backward=True)
    triton_out, triton_grad_i = call_linear(inputs, triton_linear, grad=grad, backward=True)

    torch_grad_w = torch_linear.weight.grad.clone().masked_fill_(~dense_mask[:max_out_features], 0.)
    triton_grad_w = triton_linear.weight.grad.clone()

    torch.testing.assert_close(torch_out, naive_out, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(torch_out, triton_out, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(torch_grad_i, naive_grad_i, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(torch_grad_i, triton_grad_i, atol=1e-3, rtol=1e-3)
    # torch.testing.assert_close(torch_grad_w, naive_dense_grad_w, atol=1e-2, rtol=1e-2)
    # torch.testing.assert_close(naive_sparse_grad_w, triton_grad_w, atol=1e-2, rtol=1e-2)
    torch_sparse_grad_w = triton_linear.to_sparse_weight(
        torch.cat([
            torch_grad_w,
            torch.zeros((hidden_dim - max_out_features, dim), dtype=dtype, device=device),
        ], dim=0)
    )
    torch.testing.assert_close(torch_sparse_grad_w, triton_grad_w, atol=1e-2, rtol=1e-2)

    linears = [torch_linear, naive_linear, triton_linear]
    t3 = [profile(call_linear, [inputs, linear, grad, True]) for linear in linears]
    inputs.requires_grad = False
    t2 = [profile(call_linear, [inputs, linear, grad, True]) for linear in linears]
    t3 = [y - x for x, y in zip(t2, t3)]
    t1 = [profile(call_linear, [inputs, linear, None, False]) for linear in linears]
    t2 = [y - x for x, y in zip(t1, t2)]

    import pandas as pd
    df = pd.DataFrame([t1, t2, t3], index=['fwd', 'bwd_w', 'bwd_i'], columns=['dense', 'naive', 'triton'])
    print(df.T.round(2))


if __name__ == "__main__":
    test_masked_linear(4096, 4096, 4096, [8, 8, 16])
    test_masked_linear(6789, 4096, 8192, [4] * 8, 5)
    test_masked_linear(8192, 8192, 8192, [1] * 32)
    test_masked_linear(8192, 8192, 8192, [1] * 32, 15)
