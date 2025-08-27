import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_loop(
    q, k_ptrs, v_ptrs, o, m, l,
    offs_m, offs_n, stride_kn, stride_vn,
    start, end, BLOCK_SIZE_N: tl.constexpr,
    CAUSAL: tl.constexpr, MASK_N: tl.constexpr,
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

        # Calc S <- Q @ K^T, mask S if causal
        s = tl.dot(q, k.T)
        if CAUSAL:
            causal_mask = offs_m[:, None] >= start_n + offs_n[None, :]
            s = tl.where(causal_mask, s, float("-inf"))
        elif MASK_N:
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


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_stages=4, num_warps=8),
    ],
    key=['num_tokens', 'num_heads', 'CAUSAL'],
)
@triton.jit
def flash_attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lb, stride_lh, stride_lm,
    sm_scale, num_tokens, num_heads, CAUSAL: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_SIZE_M
    off_h = tl.program_id(1) % num_heads
    off_b = tl.program_id(2) // num_heads

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
    if CAUSAL:
        start, mid, end = 0, start_m // BLOCK_SIZE_N * BLOCK_SIZE_N, tl.minimum(start_m + BLOCK_SIZE_M, num_tokens)
    else:
        start, mid, end = 0, num_tokens // BLOCK_SIZE_N * BLOCK_SIZE_N, num_tokens

    # Main loop part 1: no mask at first
    o, m, l = _attn_fwd_loop(
        q, k_ptrs, v_ptrs, o, m, l,
        offs_m, offs_n, stride_kn, stride_vn,
        start, mid, BLOCK_SIZE_N=BLOCK_SIZE_N,
        CAUSAL=False, MASK_N=False,
    )
    # Main loop part 2: causal mask and kv data mask for last blocks
    o, m, l = _attn_fwd_loop(
        q, k_ptrs, v_ptrs, o, m, l,
        offs_m, offs_n, stride_kn, stride_vn,
        mid, end, BLOCK_SIZE_N=BLOCK_SIZE_N,
        CAUSAL=CAUSAL, MASK_N=True,
    )

    # Write back output and log_sum_exp
    o = o / l[:, None]
    tl.store(o_ptrs, o.to(o_ptr.type.element_ty), mask=mask_m[:, None])
    lse = tl.math.log2(l) + m  # Here lse is still logged with base two
    tl.store(lse_ptrs, lse.to(lse_ptr.type.element_ty), mask=mask_m)


@triton.jit
def _attn_bwd_loop(
    k, v, dk, dv, dq_ptrs, q_ptrs, do_ptrs, lse_ptrs, delta_ptrs,
    offs_m, offs_n, stride_qm, stride_om, stride_lm, stride_dm,
    start, end, BLOCK_SIZE_M: tl.constexpr,
    CAUSAL: tl.constexpr, MASK_M: tl.constexpr, SKIP_DQ: tl.constexpr,
):
    # Reverse loop to maximize L2 cache hit rate when causal
    for start_m in range(tl.cdiv(end, BLOCK_SIZE_M) * BLOCK_SIZE_M - BLOCK_SIZE_M, start - BLOCK_SIZE_M, -BLOCK_SIZE_M):

        # Load Q, dO, LSE (= M + log(L)), Δ (= sum(dO * O))
        if MASK_M:
            mask_m = start_m + offs_m < end
            q = tl.load(q_ptrs + start_m * stride_qm, mask=mask_m[:, None], other=0.0)
            do = tl.load(do_ptrs + start_m * stride_om, mask=mask_m[:, None], other=0.0)
            lse = tl.load(lse_ptrs + start_m * stride_lm, mask=mask_m, other=float('-inf'))
            delta = tl.load(delta_ptrs + start_m * stride_dm, mask=mask_m, other=0.0)
        else:
            q = tl.load(q_ptrs + start_m * stride_qm)
            do = tl.load(do_ptrs + start_m * stride_om)
            lse = tl.load(lse_ptrs + start_m * stride_lm)
            delta = tl.load(delta_ptrs + start_m * stride_dm)

        # Calc P <- exp(Q @ K^T - M) / L
        pT = tl.math.exp2(tl.dot(k, q.T) - lse[None, :])
        if CAUSAL:
            causal_mask = start_m + offs_m[None, :] >= offs_n[:, None]
            pT = tl.where(causal_mask, pT, 0.0)
        if MASK_M:
            pT = tl.where(mask_m[None, :], pT, 0.0)

        # Update dV <- dV + P^T @ dO
        dv += tl.dot(pT.to(do.type.element_ty), do)

        # Calc dS <- P * (dO @ V^T - Δ)
        dsT = (pT * (tl.dot(v, do.T) - delta[None, :])).to(q.type.element_ty)
        if MASK_M:
            dsT = tl.where(mask_m[None, :], dsT, 0.0)

        if not SKIP_DQ:
            # Update dQ <- dQ + dS @ K by atomic_add
            dqT = tl.dot(k.T, dsT) * 0.6931471824645996
            if MASK_M:
                tl.atomic_add(dq_ptrs + start_m * stride_om, dqT, mask=mask_m[None, :], sem='relaxed')
            else:
                tl.atomic_add(dq_ptrs + start_m * stride_om, dqT, sem='relaxed')

        # Update dK <- dK + dS^T @ Q
        dk += tl.dot(dsT, q)

    return dk, dv


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_stages=1, num_warps=8),
    ],
    key=['num_tokens', 'num_heads', 'CAUSAL'],
)
@triton.jit
def flash_attn_bwd_kernel(
    q_ptr, k_ptr, v_ptr, lse_ptr, delta_ptr,
    dq_ptr, dk_ptr, dv_ptr, do_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lb, stride_lh, stride_lm,
    stride_db, stride_dh, stride_dm,
    sm_scale, num_tokens, num_heads, CAUSAL: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    SKIP_DQ: tl.constexpr,
):
    start_n = tl.program_id(0) * BLOCK_SIZE_N
    off_h = tl.program_id(1) % num_heads
    off_b = tl.program_id(2) // num_heads

    # Initialize offsets
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    mask_n = offs_n < num_tokens
    q_ptrs = q_ptr + off_b * stride_qb + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    k_ptrs = k_ptr + off_b * stride_kb + off_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    v_ptrs = v_ptr + off_b * stride_vb + off_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    dq_ptrs = dq_ptr + off_b * stride_qb + off_h * stride_qh + offs_m[None, :] * stride_qm + offs_d[:, None] * stride_qd
    dk_ptrs = dk_ptr + off_b * stride_kb + off_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    dv_ptrs = dv_ptr + off_b * stride_vb + off_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    do_ptrs = do_ptr + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    lse_ptrs = lse_ptr + off_b * stride_lb + off_h * stride_lh + offs_m * stride_lm
    delta_ptrs = delta_ptr + off_b * stride_db + off_h * stride_dh + offs_m * stride_dm

    # Initialize dK and dV
    dk = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_D], dtype=tl.float32)

    # Load K and V
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    k = (k * (sm_scale * 1.4426950408889634)).to(k_ptr.type.element_ty)

    # Split the main loop into 3 parts
    if CAUSAL:
        causal_start = start_n // BLOCK_SIZE_M * BLOCK_SIZE_M
        causal_end = tl.minimum(causal_start + tl.maximum(BLOCK_SIZE_M, BLOCK_SIZE_N), num_tokens)
        full_start = causal_end
        full_end = num_tokens // BLOCK_SIZE_M * BLOCK_SIZE_M
        mask_start = tl.maximum(full_end, causal_end)
        mask_end = num_tokens
    else:
        causal_start, causal_end = 0, 0
        full_start, full_end = 0, num_tokens // BLOCK_SIZE_M * BLOCK_SIZE_M
        mask_start, mask_end = full_end, num_tokens

    # Main loop part 3: mask last rows that exceed the sequence length
    dk, dv = _attn_bwd_loop(
        k, v, dk, dv, dq_ptrs, q_ptrs, do_ptrs, lse_ptrs, delta_ptrs,
        offs_m, offs_n, stride_qm, stride_om, stride_lm, stride_dm,
        mask_start, mask_end, BLOCK_SIZE_M=BLOCK_SIZE_M,
        CAUSAL=False, MASK_M=True, SKIP_DQ=SKIP_DQ,
    )
    # Main loop part 2: no mask
    dk, dv = _attn_bwd_loop(
        k, v, dk, dv, dq_ptrs, q_ptrs, do_ptrs, lse_ptrs, delta_ptrs,
        offs_m, offs_n, stride_qm, stride_om, stride_lm, stride_dm,
        full_start, full_end, BLOCK_SIZE_M=BLOCK_SIZE_M,
        CAUSAL=False, MASK_M=False, SKIP_DQ=SKIP_DQ,
    )
    # Main loop part 1: causal mask
    dk, dv = _attn_bwd_loop(
        k, v, dk, dv, dq_ptrs, q_ptrs, do_ptrs, lse_ptrs, delta_ptrs,
        offs_m, offs_n, stride_qm, stride_om, stride_lm, stride_dm,
        causal_start, causal_end, BLOCK_SIZE_M=BLOCK_SIZE_M,
        CAUSAL=True, MASK_M=True, SKIP_DQ=SKIP_DQ,
    )

    # Write back dK and dV
    tl.store(dk_ptrs, (dk * sm_scale).to(dk_ptr.type.element_ty), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(dv_ptr.type.element_ty), mask=mask_n[:, None])


@triton.jit
def _attn_bwd_dq_loop(
    q, k_ptrs, v_ptrs, lse, delta, dq, do,
    offs_m, offs_n, stride_kn, stride_vn,
    start, end, BLOCK_SIZE_N: tl.constexpr,
    CAUSAL: tl.constexpr, MASK_N: tl.constexpr,
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

        # Calc P <- exp(Q @ K^T - M) / L
        p = tl.math.exp2(tl.dot(q, k.T) - lse[:, None])
        if CAUSAL:
            causal_mask = offs_m[:, None] >= start_n + offs_n[None, :]
            p = tl.where(causal_mask, p, 0.0)
        elif MASK_N:
            p = tl.where(mask_n[None, :], p, 0.0)

        # Calc dS <- P * (dO @ V^T - Δ)
        ds = (p * (tl.dot(do, v.T) - delta[:, None])).to(q.type.element_ty)
        if MASK_N:
            ds = tl.where(mask_n[None, :], ds, 0.0)

        # Update dQ <- dQ + dS @ K
        dq += tl.dot(ds, k)

    return dq


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_stages=4, num_warps=8),
    ],
    key=['num_tokens', 'num_heads', 'CAUSAL'],
)
@triton.jit
def flash_attn_bwd_dq_kernel(
    q_ptr, k_ptr, v_ptr, lse_ptr, delta_ptr,
    dq_ptr, do_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lb, stride_lh, stride_lm,
    stride_db, stride_dh, stride_dm,
    sm_scale, num_tokens, num_heads, CAUSAL: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_SIZE_M
    off_h = tl.program_id(1) % num_heads
    off_b = tl.program_id(2) // num_heads

    # Initialize offsets
    offs_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    mask_m = offs_m < num_tokens
    q_ptrs = q_ptr + off_b * stride_qb + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    k_ptrs = k_ptr + off_b * stride_kb + off_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    v_ptrs = v_ptr + off_b * stride_vb + off_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    dq_ptrs = dq_ptr + off_b * stride_qb + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    do_ptrs = do_ptr + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    lse_ptrs = lse_ptr + off_b * stride_lb + off_h * stride_lh + offs_m * stride_lm
    delta_ptrs = delta_ptr + off_b * stride_db + off_h * stride_dh + offs_m * stride_dm

    # Initialize dQ
    dq = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_D], dtype=tl.float32)

    # Load Q, dO, LSE (= M + log(L)), Δ (= sum(dO * O))
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    lse = tl.load(lse_ptrs, mask=mask_m, other=0.0)
    delta = tl.load(delta_ptrs, mask=mask_m, other=0.0)
    q = (q * (sm_scale * 1.4426950408889634)).to(q_ptr.type.element_ty)

    # Split the main loop into 3 parts
    if CAUSAL:
        start, mid, end = 0, start_m // BLOCK_SIZE_N * BLOCK_SIZE_N, tl.minimum(start_m + BLOCK_SIZE_M, num_tokens)
    else:
        start, mid, end = 0, num_tokens // BLOCK_SIZE_N * BLOCK_SIZE_N, num_tokens

    # Main loop part 1: no mask at first
    dq = _attn_bwd_dq_loop(
        q, k_ptrs, v_ptrs, lse, delta, dq, do,
        offs_m, offs_n, stride_kn, stride_vn,
        start, mid, BLOCK_SIZE_N=BLOCK_SIZE_N,
        CAUSAL=False, MASK_N=False,
    )
    # Main loop part 2: causal mask and kv data mask for last blocks
    dq = _attn_bwd_dq_loop(
        q, k_ptrs, v_ptrs, lse, delta, dq, do,
        offs_m, offs_n, stride_kn, stride_vn,
        mid, end, BLOCK_SIZE_N=BLOCK_SIZE_N,
        CAUSAL=CAUSAL, MASK_N=True,
    )

    # Write back dQ
    tl.store(dq_ptrs, (dq * sm_scale).to(dq_ptr.type.element_ty), mask=mask_m[:, None])


@torch.compile
def flash_attn_bwd_calc_delta(
    o: torch.Tensor,   # [batch_size, num_tokens, num_heads, head_dim]
    do: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
):
    return torch.sum((o * do).swapaxes(1, 2), dim=-1, dtype=torch.float32)


class TritonFlashAttention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sm_scale: float,
        causal: bool,
        split_bwd: bool,
    ):
        batch_size, num_tokens, num_heads, head_dim = q.shape
        batch_size_k, num_k_tokens, num_k_heads, head_dim_k = k.shape
        batch_size_v, num_v_tokens, num_v_heads, head_dim_v = v.shape
        assert batch_size == batch_size_k and batch_size_k == batch_size_v
        assert num_tokens == num_k_tokens and num_k_tokens == num_v_tokens
        assert num_heads == num_k_heads and num_k_heads == num_v_heads
        assert head_dim == head_dim_k and head_dim_k == head_dim_v
        assert head_dim in {16, 32, 64, 128}

        sm_scale = sm_scale or head_dim ** (-0.5)
        o = torch.empty_like(q)
        lse = torch.empty((batch_size, num_heads, num_tokens), dtype=torch.float32, device=q.device)

        grid = lambda META: (triton.cdiv(num_tokens, META['BLOCK_SIZE_M']), num_heads * batch_size, 1)
        flash_attn_fwd_kernel[grid](
            q, k, v, o, lse,
            q.stride(0), q.stride(2), q.stride(1), q.stride(3),
            k.stride(0), k.stride(2), k.stride(1), k.stride(3),
            v.stride(0), v.stride(2), v.stride(1), v.stride(3),
            o.stride(0), o.stride(2), o.stride(1), o.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            sm_scale, num_tokens, num_heads,
            CAUSAL=causal, BLOCK_SIZE_D=head_dim,
        )

        ctx.batch_size = batch_size
        ctx.num_heads = num_heads
        ctx.num_tokens = num_tokens
        ctx.head_dim = head_dim
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.split_bwd = split_bwd
        ctx.save_for_backward(q, k, v, o, lse)

        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        q, k, v, o, lse = ctx.saved_tensors

        delta = flash_attn_bwd_calc_delta(o, do)
        if ctx.split_bwd:
            dq = torch.empty_like(q)
        else:
            dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        grid = lambda META: (triton.cdiv(ctx.num_tokens, META['BLOCK_SIZE_N']), ctx.num_heads * ctx.batch_size, 1)
        flash_attn_bwd_kernel[grid](
            q, k, v, lse, delta, dq, dk, dv, do,
            q.stride(0), q.stride(2), q.stride(1), q.stride(3),
            k.stride(0), k.stride(2), k.stride(1), k.stride(3),
            v.stride(0), v.stride(2), v.stride(1), v.stride(3),
            o.stride(0), o.stride(2), o.stride(1), o.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            ctx.sm_scale, ctx.num_tokens, ctx.num_heads,
            CAUSAL=ctx.causal, BLOCK_SIZE_D=ctx.head_dim,
            SKIP_DQ=ctx.split_bwd,
        )

        if ctx.split_bwd:
            grid = lambda META: (triton.cdiv(ctx.num_tokens, META['BLOCK_SIZE_M']), ctx.num_heads * ctx.batch_size, 1)
            flash_attn_bwd_dq_kernel[grid](
                q, k, v, lse, delta, dq, do,
                q.stride(0), q.stride(2), q.stride(1), q.stride(3),
                k.stride(0), k.stride(2), k.stride(1), k.stride(3),
                v.stride(0), v.stride(2), v.stride(1), v.stride(3),
                o.stride(0), o.stride(2), o.stride(1), o.stride(3),
                lse.stride(0), lse.stride(1), lse.stride(2),
                delta.stride(0), delta.stride(1), delta.stride(2),
                ctx.sm_scale, ctx.num_tokens, ctx.num_heads,
                CAUSAL=ctx.causal, BLOCK_SIZE_D=ctx.head_dim,
            )
        else:
            dq = dq.to(q.dtype)

        return dq, dk, dv, None, None, None


def triton_flash_attn_func(
    q: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    softmax_scale: float = None,
    causal: bool = True,
):
    return TritonFlashAttention.apply(q, k, v, softmax_scale, causal, False)


def triton_flash_attn_split_bwd_func(
    q: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    softmax_scale: float = None,
    causal: bool = True,
):
    return TritonFlashAttention.apply(q, k, v, softmax_scale, causal, True)
