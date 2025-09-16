import torch
import triton
import triton.language as tl
import pandas as pd
from flash_attn import flash_attn_func


@triton.jit
def _attn_fwd_loop(
    q, k_ptrs, v_ptrs, o, m, l,
    offs_m, offs_n, stride_kn, stride_vn,
    start, end, BLOCK_SIZE_N: tl.constexpr,
    MASK_N: tl.constexpr,
):
    for start_n in range(start, end, BLOCK_SIZE_N):

        # Load K, V
        if MASK_N:
            mask_n = start_n + offs_n < end
            k = tl.load(k_ptrs + start_n * stride_kn, mask=mask_n[:, None], other=0.0)
            v = tl.load(v_ptrs + start_n * stride_vn, mask=mask_n[:, None], other=0.0)
        else:
            k = tl.load(k_ptrs + start_n * stride_kn)
            v = tl.load(v_ptrs + start_n * stride_vn)

        # Calc S <- Q @ K^T
        s = tl.dot(q, k.T)
        if MASK_N:
            s = tl.where(mask_n[None, :], s, float("-inf"))

        # Calc new row max M' <- max(M, max(s)), P <- exp(S - M'), local sum exp L1 <- sum(P)
        m_new = tl.maximum(m, tl.max(s, 1))
        p = tl.math.exp2(s - m_new[:, None])
        l_local = tl.sum(p, 1)

        # Update L <- L * exp(M - M') + L1, M <- M'
        alpha = tl.math.exp2(m - m_new)
        l = l * alpha + l_local
        m = m_new

        # Update O <- O * exp(M - M') + P @ V
        o = o * alpha[:, None] + tl.dot(p.to(v.type.element_ty), v)

    return o, m, l


@triton.jit
def _block_sparse_attn_fwd_loop(
    q, k_ptrs, v_ptrs, o, m, l, idx_ptr, num_tokens,
    offs_m, offs_n, stride_kn, stride_vn,
    start, end, BLOCK_SIZE_N: tl.constexpr,
    MASK_N: tl.constexpr,
):
    for j in range(start, end):

        # Load K, V
        start_n = tl.load(idx_ptr + j) * BLOCK_SIZE_N
        if MASK_N:
            mask_n = start_n + offs_n < num_tokens
            k = tl.load(k_ptrs + start_n * stride_kn, mask=mask_n[:, None], other=0.0)
            v = tl.load(v_ptrs + start_n * stride_vn, mask=mask_n[:, None], other=0.0)
        else:
            k = tl.load(k_ptrs + start_n * stride_kn)
            v = tl.load(v_ptrs + start_n * stride_vn)

        # Calc S <- Q @ K^T
        s = tl.dot(q, k.T)
        if MASK_N:
            s = tl.where(mask_n[None, :], s, float("-inf"))

        # Calc new row max M' <- max(M, max(s)), P <- exp(S - M'), local sum exp L1 <- sum(P)
        m_new = tl.maximum(m, tl.max(s, 1))
        p = tl.math.exp2(s - m_new[:, None])
        l_local = tl.sum(p, 1)

        # Update L <- L * exp(M - M') + L1, M <- M'
        alpha = tl.math.exp2(m - m_new)
        l = l * alpha + l_local
        m = m_new

        # Update O <- O * exp(M - M') + P @ V
        o = o * alpha[:, None] + tl.dot(p.to(v.type.element_ty), v)

    return o, m, l


@triton.jit
def _pit_sparse_attn_fwd_loop(
    q, k_ptrs, v_ptrs, o, m, l, idx_ptr,
    offs_m, offs_n, stride_kn, stride_vn,
    start, end, BLOCK_SIZE_N: tl.constexpr,
    MASK_N: tl.constexpr,
):
    for start_n in range(start, end, BLOCK_SIZE_N):

        # Load K, V
        if MASK_N:
            mask_n = start_n + offs_n < end
            idx = tl.load(idx_ptr + start_n + offs_n, mask=mask_n, other=0)
            k = tl.load(k_ptrs + idx[:, None] * stride_kn, mask=mask_n[:, None], other=0.0)
            v = tl.load(v_ptrs + idx[:, None] * stride_vn, mask=mask_n[:, None], other=0.0)
        else:
            idx = tl.load(idx_ptr + start_n + offs_n)
            k = tl.load(k_ptrs + idx[:, None] * stride_kn)
            v = tl.load(v_ptrs + idx[:, None] * stride_vn)

        # Calc S <- Q @ K^T
        s = tl.dot(q, k.T)
        if MASK_N:
            s = tl.where(mask_n[None, :], s, float("-inf"))

        # Calc new row max M' <- max(M, max(s)), P <- exp(S - M'), local sum exp L1 <- sum(P)
        m_new = tl.maximum(m, tl.max(s, 1))
        p = tl.math.exp2(s - m_new[:, None])
        l_local = tl.sum(p, 1)

        # Update L <- L * exp(M - M') + L1, M <- M'
        alpha = tl.math.exp2(m - m_new)
        l = l * alpha + l_local
        m = m_new

        # Update O <- O * exp(M - M') + P @ V
        o = o * alpha[:, None] + tl.dot(p.to(v.type.element_ty), v)

    return o, m, l


@triton.jit
def flash_attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lb, stride_lh, stride_lm,
    sm_scale, num_tokens, num_heads,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_SIZE_M
    off_h = tl.program_id(1) % num_heads
    off_b = tl.program_id(1) // num_heads

    # Initialize offsets
    offs_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    mask_m = offs_m < num_tokens
    q_ptrs = q_ptr + off_b * stride_qb + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    k_ptrs = k_ptr + off_b * stride_kb + off_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    v_ptrs = v_ptr + off_b * stride_vb + off_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    o_ptrs = o_ptr + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    lse_ptrs = lse_ptr + off_b * stride_lb + off_h * stride_lh + offs_m * stride_lm

    # Initialize row_max (M) and sum_exp (L)
    m = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) - float("inf")
    l = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    o = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_D], dtype=tl.float32)

    # Scale sm_scale by log_2(e) and use 2^x instead of exp
    sm_scale = sm_scale * 1.4426950408889634

    # Load Q: it will stay in SRAM throughout
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    q = (q * sm_scale).to(q_ptr.type.element_ty)

    # Split the main loop into 2 parts
    start, mid, end = 0, num_tokens // BLOCK_SIZE_N * BLOCK_SIZE_N, num_tokens

    # Main loop part 1: no mask at first
    o, m, l = _attn_fwd_loop(
        q, k_ptrs, v_ptrs, o, m, l,
        offs_m, offs_n, stride_kn, stride_vn,
        start, mid, BLOCK_SIZE_N=BLOCK_SIZE_N,
        MASK_N=False,
    )
    # Main loop part 2: kv data mask for the last block
    o, m, l = _attn_fwd_loop(
        q, k_ptrs, v_ptrs, o, m, l,
        offs_m, offs_n, stride_kn, stride_vn,
        mid, end, BLOCK_SIZE_N=BLOCK_SIZE_N,
        MASK_N=True,
    )

    # Write back output and log_sum_exp
    o = o / l[:, None]
    tl.store(o_ptrs, o.to(o_ptr.type.element_ty), mask=mask_m[:, None])
    lse = tl.math.log2(l) + m  # Here lse is still logged with base two
    tl.store(lse_ptrs, lse.to(lse_ptr.type.element_ty), mask=mask_m)


@triton.jit
def block_sparse_flash_attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
    cnt_ptr, idx_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lb, stride_lh, stride_lm,
    stride_cb, stride_ch, stride_cm,
    stride_ib, stride_ih, stride_im,
    sm_scale, num_tokens, num_heads,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1) % num_heads
    off_b = tl.program_id(1) // num_heads

    # Initialize offsets
    offs_m = start_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    mask_m = offs_m < num_tokens
    q_ptrs = q_ptr + off_b * stride_qb + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    k_ptrs = k_ptr + off_b * stride_kb + off_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    v_ptrs = v_ptr + off_b * stride_vb + off_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    o_ptrs = o_ptr + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    lse_ptrs = lse_ptr + off_b * stride_lb + off_h * stride_lh + offs_m * stride_lm
    cnt_ptr = cnt_ptr + off_b * stride_cb + off_h * stride_ch + start_m * stride_cm
    idx_ptr = idx_ptr + off_b * stride_ib + off_h * stride_ih + start_m * stride_im

    # Initialize row_max (M) and sum_exp (L)
    m = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) - float("inf")
    l = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    o = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_D], dtype=tl.float32)

    # Scale sm_scale by log_2(e) and use 2^x instead of exp
    sm_scale = sm_scale * 1.4426950408889634

    # Load Q: it will stay in SRAM throughout
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    q = (q * sm_scale).to(q_ptr.type.element_ty)

    # Split the main loop into 2 parts
    block_cnt = tl.load(cnt_ptr)
    start, mid, end = 0, block_cnt - 1, block_cnt

    # Main loop part 1: no mask at first
    o, m, l = _block_sparse_attn_fwd_loop(
        q, k_ptrs, v_ptrs, o, m, l, idx_ptr, num_tokens,
        offs_m, offs_n, stride_kn, stride_vn,
        start, mid, BLOCK_SIZE_N=BLOCK_SIZE_N,
        MASK_N=False,
    )
    # Main loop part 2: kv data mask for the last block
    o, m, l = _block_sparse_attn_fwd_loop(
        q, k_ptrs, v_ptrs, o, m, l, idx_ptr, num_tokens,
        offs_m, offs_n, stride_kn, stride_vn,
        mid, end, BLOCK_SIZE_N=BLOCK_SIZE_N,
        MASK_N=True,
    )

    # Write back output and log_sum_exp
    o = o / l[:, None]
    tl.store(o_ptrs, o.to(o_ptr.type.element_ty), mask=mask_m[:, None])
    lse = tl.math.log2(l) + m  # Here lse is still logged with base two
    tl.store(lse_ptrs, lse.to(lse_ptr.type.element_ty), mask=mask_m)


@triton.jit
def pit_sparse_flash_attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
    cnt_ptr, idx_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lb, stride_lh, stride_lm,
    stride_cb, stride_ch, stride_cm,
    stride_ib, stride_ih, stride_im,
    sm_scale, num_tokens, num_heads,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1) % num_heads
    off_b = tl.program_id(1) // num_heads

    # Initialize offsets
    offs_m = start_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    mask_m = offs_m < num_tokens
    q_ptrs = q_ptr + off_b * stride_qb + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    k_ptrs = k_ptr + off_b * stride_kb + off_h * stride_kh + offs_d[None, :] * stride_kd
    v_ptrs = v_ptr + off_b * stride_vb + off_h * stride_vh + offs_d[None, :] * stride_vd
    o_ptrs = o_ptr + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    lse_ptrs = lse_ptr + off_b * stride_lb + off_h * stride_lh + offs_m * stride_lm
    cnt_ptr = cnt_ptr + off_b * stride_cb + off_h * stride_ch + start_m * stride_cm
    idx_ptr = idx_ptr + off_b * stride_ib + off_h * stride_ih + start_m * stride_im

    # Initialize row_max (M) and sum_exp (L)
    m = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) - float("inf")
    l = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    o = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_D], dtype=tl.float32)

    # Scale sm_scale by log_2(e) and use 2^x instead of exp
    sm_scale = sm_scale * 1.4426950408889634

    # Load Q: it will stay in SRAM throughout
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    q = (q * sm_scale).to(q_ptr.type.element_ty)

    # Split the main loop into 2 parts
    num_selected_tokens = tl.load(cnt_ptr)
    start, mid, end = 0, num_selected_tokens // BLOCK_SIZE_N * BLOCK_SIZE_N, num_selected_tokens

    # Main loop part 1: no mask at first
    o, m, l = _pit_sparse_attn_fwd_loop(
        q, k_ptrs, v_ptrs, o, m, l, idx_ptr,
        offs_m, offs_n, stride_kn, stride_vn,
        start, mid, BLOCK_SIZE_N=BLOCK_SIZE_N,
        MASK_N=False,
    )
    # Main loop part 2: kv data mask for the last block
    o, m, l = _pit_sparse_attn_fwd_loop(
        q, k_ptrs, v_ptrs, o, m, l, idx_ptr,
        offs_m, offs_n, stride_kn, stride_vn,
        mid, end, BLOCK_SIZE_N=BLOCK_SIZE_N,
        MASK_N=True,
    )

    # Write back output and log_sum_exp
    o = o / l[:, None]
    tl.store(o_ptrs, o.to(o_ptr.type.element_ty), mask=mask_m[:, None])
    lse = tl.math.log2(l) + m  # Here lse is still logged with base two
    tl.store(lse_ptrs, lse.to(lse_ptr.type.element_ty), mask=mask_m)


def triton_flash_attn_func(
    q: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    softmax_scale: float = None,
):
    batch_size, num_tokens, num_heads, head_dim = q.shape
    batch_size_k, num_k_tokens, num_k_heads, head_dim_k = k.shape
    batch_size_v, num_v_tokens, num_v_heads, head_dim_v = v.shape
    assert batch_size == batch_size_k and batch_size_k == batch_size_v
    assert num_tokens == num_k_tokens and num_k_tokens == num_v_tokens
    assert num_heads == num_k_heads and num_k_heads == num_v_heads
    assert head_dim == head_dim_k and head_dim_k == head_dim_v
    assert head_dim in {16, 32, 64, 128}

    block_size_M = 128
    block_size_N = 32
    num_warps = 4
    num_stages = 4

    sm_scale = softmax_scale or head_dim ** (-0.5)
    o = torch.empty_like(q)
    lse = torch.empty((batch_size, num_heads, num_tokens), dtype=torch.float32, device=q.device)

    grid = (triton.cdiv(num_tokens, block_size_M), num_heads * batch_size, 1)
    flash_attn_fwd_kernel[grid](
        q, k, v, o, lse,
        q.stride(0), q.stride(2), q.stride(1), q.stride(3),
        k.stride(0), k.stride(2), k.stride(1), k.stride(3),
        v.stride(0), v.stride(2), v.stride(1), v.stride(3),
        o.stride(0), o.stride(2), o.stride(1), o.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        sm_scale, num_tokens, num_heads,
        BLOCK_SIZE_M=block_size_M, BLOCK_SIZE_N=block_size_N, BLOCK_SIZE_D=head_dim,
        num_warps=num_warps, num_stages=num_stages,
    )
    return o


def triton_block_sparse_flash_attn_func(
    q: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    block_cnt: torch.Tensor,  # [batch_size, num_heads, num_tokens // block_size_M]
    block_idx: torch.Tensor,  # [batch_size, num_heads, num_tokens // block_size_M, num_tokens // block_size_N]
    block_size_M: int,
    block_size_N: int,
    softmax_scale: float = None,
):
    batch_size, num_tokens, num_heads, head_dim = q.shape
    batch_size_k, num_k_tokens, num_k_heads, head_dim_k = k.shape
    batch_size_v, num_v_tokens, num_v_heads, head_dim_v = v.shape
    assert batch_size == batch_size_k and batch_size_k == batch_size_v
    assert num_tokens == num_k_tokens and num_k_tokens == num_v_tokens
    assert num_heads == num_k_heads and num_k_heads == num_v_heads
    assert head_dim == head_dim_k and head_dim_k == head_dim_v
    assert head_dim in {16, 32, 64, 128}

    num_warps = 4
    num_stages = 4

    sm_scale = softmax_scale or head_dim ** (-0.5)
    o = torch.empty_like(q)
    lse = torch.empty((batch_size, num_heads, num_tokens), dtype=torch.float32, device=q.device)

    grid = (triton.cdiv(num_tokens, block_size_M), num_heads * batch_size, 1)
    block_sparse_flash_attn_fwd_kernel[grid](
        q, k, v, o, lse, block_cnt, block_idx,
        q.stride(0), q.stride(2), q.stride(1), q.stride(3),
        k.stride(0), k.stride(2), k.stride(1), k.stride(3),
        v.stride(0), v.stride(2), v.stride(1), v.stride(3),
        o.stride(0), o.stride(2), o.stride(1), o.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        block_cnt.stride(0), block_cnt.stride(1), block_cnt.stride(2),
        block_idx.stride(0), block_idx.stride(1), block_idx.stride(2),
        sm_scale, num_tokens, num_heads,
        BLOCK_SIZE_M=block_size_M, BLOCK_SIZE_N=block_size_N, BLOCK_SIZE_D=head_dim,
        num_warps=num_warps, num_stages=num_stages,
    )
    return o


def triton_pit_sparse_flash_attn_func(
    q: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    column_cnt: torch.Tensor,  # [batch_size, num_heads, num_tokens // block_size_M]
    column_idx: torch.Tensor,  # [batch_size, num_heads, num_tokens // block_size_M, num_tokens // block_size_N]
    block_size_M: int,
    block_size_N: int,
    softmax_scale: float = None,
):
    batch_size, num_tokens, num_heads, head_dim = q.shape
    batch_size_k, num_k_tokens, num_k_heads, head_dim_k = k.shape
    batch_size_v, num_v_tokens, num_v_heads, head_dim_v = v.shape
    assert batch_size == batch_size_k and batch_size_k == batch_size_v
    assert num_tokens == num_k_tokens and num_k_tokens == num_v_tokens
    assert num_heads == num_k_heads and num_k_heads == num_v_heads
    assert head_dim == head_dim_k and head_dim_k == head_dim_v
    assert head_dim in {16, 32, 64, 128}

    num_warps = 4
    num_stages = 4

    sm_scale = softmax_scale or head_dim ** (-0.5)
    o = torch.empty_like(q)
    lse = torch.empty((batch_size, num_heads, num_tokens), dtype=torch.float32, device=q.device)

    grid = (triton.cdiv(num_tokens, block_size_M), num_heads * batch_size, 1)
    pit_sparse_flash_attn_fwd_kernel[grid](
        q, k, v, o, lse, column_cnt, column_idx,
        q.stride(0), q.stride(2), q.stride(1), q.stride(3),
        k.stride(0), k.stride(2), k.stride(1), k.stride(3),
        v.stride(0), v.stride(2), v.stride(1), v.stride(3),
        o.stride(0), o.stride(2), o.stride(1), o.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        column_cnt.stride(0), column_cnt.stride(1), column_cnt.stride(2),
        column_idx.stride(0), column_idx.stride(1), column_idx.stride(2),
        sm_scale, num_tokens, num_heads,
        BLOCK_SIZE_M=block_size_M, BLOCK_SIZE_N=block_size_N, BLOCK_SIZE_D=head_dim,
        num_warps=num_warps, num_stages=num_stages,
    )
    return o


def torch_naive_attn(
    q: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    softmax_scale: float = None,
    mask: torch.Tensor = None,  # [batch_size, num_heads, num_tokens, num_tokens]
):
    batch_size, num_tokens, num_heads, head_dim = q.shape
    softmax_scale = softmax_scale or head_dim ** (-0.5)
    s = torch.einsum('bmhd, bnhd -> bhmn', q, k).to(torch.float32) * softmax_scale
    if mask is not None:
        s.masked_fill_(mask, float('-inf'))
    p = torch.softmax(s, dim=-1).to(q.dtype)
    o = torch.einsum('bhmn, bnhd -> bmhd', p, v)
    return o


def calc_tflops(latency: float, B: int, N: int, H: int, D: int, sparsity: float = 0.0):
    flops = 2 * 2 * B * H * N * N * D * (1.0 - sparsity)
    return (flops / 1e12) / (latency / 1e3)


def test_sparse_attn(B: int, N: int, H: int, D: int, block_size_M: int = 128, block_size_N: int = 32):
    torch.manual_seed(4321)

    q = torch.empty((B, N, H, D), dtype=torch.bfloat16, device='cuda').normal_(mean=0.0, std=0.5).requires_grad_()
    k = torch.empty((B, N, H, D), dtype=torch.bfloat16, device='cuda').normal_(mean=0.0, std=0.5).requires_grad_()
    v = torch.empty((B, N, H, D), dtype=torch.bfloat16, device='cuda').normal_(mean=0.0, std=0.5).requires_grad_()
    sm_scale = D ** (-0.5)

    num_blocks_M = triton.cdiv(N, block_size_M)
    num_blocks_N = triton.cdiv(N, block_size_N)
    block_mask = torch.randint(0, 2, (B, H, num_blocks_M, num_blocks_N), dtype=torch.bool, device='cuda')
    column_mask = torch.randint(0, 2, (B, H, num_blocks_M, N), dtype=torch.bool, device='cuda')

    full_inputs = {'q': q, 'k': k, 'v': v, 'softmax_scale': sm_scale}
    o_torch_full = torch_naive_attn(**full_inputs)
    o_flash_full = flash_attn_func(**full_inputs)
    o_triton_full = triton_flash_attn_func(**full_inputs)

    torch.testing.assert_close(o_triton_full, o_torch_full, atol=5e-4, rtol=1e-3)
    torch.testing.assert_close(o_flash_full, o_torch_full, atol=5e-4, rtol=1e-3)

    block_full_mask = torch.tile(block_mask[:, :, :, None, :, None], (1, 1, 1, block_size_M, 1, block_size_N))
    block_full_mask = block_full_mask.reshape(B, H, num_blocks_M * block_size_M, num_blocks_N * block_size_N)[:, :, :N, :N]
    block_sparsity = 1.0 - block_full_mask.sum(dtype=torch.float64).item() / (B * H * N * N)
    o_torch_block = torch_naive_attn(q, k, v, softmax_scale=sm_scale, mask=~block_full_mask)

    _, block_idx = block_mask.sort(dim=-1, stable=True, descending=True)
    block_cnt = block_mask.sum(dim=-1, dtype=torch.int32)
    block_sparse_inputs = {
        **full_inputs,
        'block_cnt': block_cnt, 'block_idx': block_idx,
        'block_size_M': block_size_M, 'block_size_N': block_size_N,
    }
    o_triton_block = triton_block_sparse_flash_attn_func(**block_sparse_inputs)

    torch.testing.assert_close(o_triton_block, o_torch_block, atol=5e-4, rtol=1e-3)

    pit_full_mask = torch.tile(column_mask[:, :, :, None, :], (1, 1, 1, block_size_M, 1))
    pit_full_mask = pit_full_mask.reshape(B, H, num_blocks_M * block_size_M, N)[:, :, :N, :]
    pit_sparsity = 1.0 - block_full_mask.sum(dtype=torch.float64).item() / (B * H * N * N)
    o_torch_pit = torch_naive_attn(q, k, v, softmax_scale=sm_scale, mask=~pit_full_mask)

    _, column_idx = column_mask.sort(dim=-1, stable=True, descending=True)
    column_cnt = column_mask.sum(dim=-1, dtype=torch.int32)
    pit_sparse_inputs = {
        **full_inputs,
        'column_cnt': column_cnt, 'column_idx': column_idx.to(torch.int32),
        'block_size_M': block_size_M, 'block_size_N': block_size_N,
    }
    o_triton_pit = triton_pit_sparse_flash_attn_func(**pit_sparse_inputs)

    torch.testing.assert_close(o_triton_pit, o_torch_pit, atol=5e-4, rtol=1e-3)

    def profile_attn_fwd(attn_func, inputs, sparsity: float = 0.0):
        latency = triton.testing.do_bench(lambda: attn_func(**inputs))
        tflops = calc_tflops(latency, B, N, H, D, sparsity)
        return {'latency': latency, 'tflops': tflops}

    print(f'Problem size: ({B}, {N}, {H}, {D})')
    print(pd.DataFrame({
        'flash(full)': profile_attn_fwd(flash_attn_func, full_inputs),
        'triton(full)': profile_attn_fwd(triton_flash_attn_func, full_inputs),
        'triton(block)': profile_attn_fwd(triton_block_sparse_flash_attn_func, block_sparse_inputs, sparsity=block_sparsity),
        'triton(pit)': profile_attn_fwd(triton_pit_sparse_flash_attn_func, pit_sparse_inputs, sparsity=pit_sparsity),
    }).T)


if __name__ == '__main__':
    test_sparse_attn(B=1, H=32, N=4321, D=128)
