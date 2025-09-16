import torch
import triton
import triton.language as tl
import pandas as pd


@triton.jit
def _moe_routing_kernel(
    a_ptr,  # [M, K]
    b_ptr,  # [E, K]
    c_ptr,  # [M, E]
    w_ptr,  # [M, T]
    t_ptr,  # [M, T]
    cnt_ptr,  # [E] in [0, M)
    idx_ptr,  # [E, M] in [0, T * M)
    M, E, K, T,
    stride_am, stride_ak,
    stride_be, stride_bk,
    stride_cm, stride_ce,
    stride_wm, stride_we,
    stride_tm, stride_te,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_E: tl.constexpr, BLOCK_SIZE_T: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)  # pid_e = 0

    if pid_m * BLOCK_SIZE_M >= M:
        return

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_e = tl.arange(0, BLOCK_SIZE_E)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_e[None, :] * stride_be + offs_k[:, None] * stride_bk

    # GeMM
    accumulator = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_E], dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Softmax
    c_max = tl.max(accumulator, axis=1)
    c_exp = tl.math.exp(accumulator - c_max[:, None])
    c_sum = tl.sum(c_exp, axis=1)
    c = c_exp / c_sum[:, None]

    # Save Routing Weights
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < M
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_e[None, :] * stride_ce
    tl.store(c_ptrs, c.to(c_ptr.type.element_ty), mask=mask_m[:, None])

    # Top-K
    w_ptrs = w_ptr + offs_m * stride_wm
    t_ptrs = t_ptr + offs_m * stride_tm
    offs_t = tl.arange(0, BLOCK_SIZE_T)
    e_idx = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_T], dtype=tl.int32)
    for k in tl.static_range(BLOCK_SIZE_T):
        max_c, max_idx = tl.max(c, axis=1, return_indices=True)
        c = tl.where(offs_e[None, :] == max_idx[:, None], 0.0, c)
        e_idx = tl.where(offs_t[None, :] == k, max_idx[:, None], e_idx)
        tl.store(w_ptrs, max_c.to(w_ptr.type.element_ty), mask=mask_m)
        tl.store(t_ptrs, max_idx.to(t_ptr.type.element_ty), mask=mask_m)
        w_ptrs += stride_we
        t_ptrs += stride_te

    # Histogram
    e_idx = tl.reshape(e_idx, [BLOCK_SIZE_M * BLOCK_SIZE_T, ])
    mask = tl.reshape(tl.broadcast_to(offs_m[:, None] < M, [BLOCK_SIZE_M, BLOCK_SIZE_T]), [BLOCK_SIZE_M * BLOCK_SIZE_T, ])
    m_idx = tl.atomic_add(cnt_ptr + e_idx, tl.zeros_like(e_idx) + 1, mask=mask, sem='relaxed')
    token_idx = tl.reshape(offs_m[:, None] * T + offs_t[None, :], [BLOCK_SIZE_M * BLOCK_SIZE_T, ])
    tl.store(idx_ptr + e_idx * M + m_idx, token_idx, mask=mask)


def fused_routing(
    hidden_states: torch.Tensor,
    gating_weight: torch.Tensor,
    num_experts: int,
    top_k: int,
):
    num_tokens, hidden_dim = hidden_states.shape
    assert gating_weight.shape == (num_experts, hidden_dim)
    router_logits = torch.empty((num_tokens, num_experts), dtype=hidden_states.dtype, device=hidden_states.device)
    routing_weights = torch.empty((num_tokens, top_k), dtype=hidden_states.dtype, device=hidden_states.device)
    selected_experts = torch.empty((num_tokens, top_k), dtype=torch.int64, device=hidden_states.device)
    expert_cnt = torch.zeros((num_experts, ), dtype=torch.int32, device=hidden_states.device)
    expert_idx = torch.zeros((num_experts, num_tokens), dtype=torch.int32, device=hidden_states.device)
    assert num_experts & (num_experts - 1) == 0
    assert top_k & (top_k - 1) == 0

    META = {
        'BLOCK_SIZE_M': 128,
        'BLOCK_SIZE_K': 32,
        'num_warps': 8,
        'num_stages': 4,
    }
    _moe_routing_kernel[triton.cdiv(num_tokens, META['BLOCK_SIZE_M']), 1, 1](
        hidden_states, gating_weight, router_logits,
        routing_weights, selected_experts, expert_cnt, expert_idx,
        num_tokens, num_experts, hidden_dim, top_k,
        hidden_states.stride(0), hidden_states.stride(1),
        gating_weight.stride(0), gating_weight.stride(1),
        router_logits.stride(0), router_logits.stride(1),
        routing_weights.stride(0), routing_weights.stride(1),
        selected_experts.stride(0), selected_experts.stride(1),
        BLOCK_SIZE_E=num_experts, BLOCK_SIZE_T=top_k,
        **META,
    )
    return router_logits, routing_weights, expert_cnt, expert_idx, selected_experts


def naive_routing(
    hidden_states: torch.Tensor,
    gating_weight: torch.Tensor,
    num_experts: int,
    top_k: int,
    selected_experts: torch.Tensor = None,
):
    num_tokens, hidden_dim = hidden_states.shape
    router_logits = torch.nn.functional.linear(hidden_states, gating_weight, bias=None)
    router_logits = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float32)
    if selected_experts is None:
        routing_weights, selected_experts = torch.topk(router_logits, top_k, dim=-1)
    else:
        assert selected_experts.shape == (num_tokens, top_k)
        routing_weights = torch.gather(router_logits, dim=1, index=selected_experts)
    routing_weights = routing_weights.to(hidden_states.dtype)
    expert_mask = torch.nn.functional.one_hot(selected_experts.flatten(), num_classes=num_experts)
    expert_cnt = expert_mask.sum(dim=0)
    expert_idx = expert_mask.argsort(dim=0, descending=True)[:num_tokens].T.contiguous()
    return router_logits, routing_weights, expert_cnt, expert_idx, selected_experts


@torch.compile
def compiled_routing(
    hidden_states: torch.Tensor,
    gating_weight: torch.Tensor,
    num_experts: int,
    top_k: int,
):
    num_tokens, hidden_dim = hidden_states.shape
    router_logits = torch.nn.functional.linear(hidden_states, gating_weight, bias=None)
    router_logits = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float32)
    routing_weights, selected_experts = torch.topk(router_logits, top_k, dim=-1)
    routing_weights = routing_weights.to(hidden_states.dtype)
    expert_mask = torch.nn.functional.one_hot(selected_experts.flatten(), num_classes=num_experts)
    expert_cnt = expert_mask.sum(dim=0)
    expert_idx = expert_mask.argsort(dim=0, descending=True)[:num_tokens].T.contiguous()
    return router_logits, routing_weights, expert_cnt, expert_idx, selected_experts


def test_routing(num_tokens: int, hidden_dim: int, num_experts: int, top_k: int):
    torch.manual_seed(4321)
    inputs = torch.randn((num_tokens, hidden_dim), dtype=torch.bfloat16, device='cuda')
    gating_weight = torch.empty((num_experts, hidden_dim), dtype=torch.bfloat16, device='cuda').normal_(mean=0.0, std=0.02)

    l1, w1, c1, i1, t1 = fused_routing(inputs, gating_weight, num_experts, top_k)
    l0, w0, c0, i0, t0 = naive_routing(inputs, gating_weight, num_experts, top_k, selected_experts=t1)

    torch.testing.assert_close(w0, w1, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(l0.to(l1.dtype), l1, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(c0.to(c1.dtype), c1)
    for i in range(num_experts):
        num_selected_tokens = c0[i].item()
        assert set(i0[i, :num_selected_tokens].cpu().tolist()) == set(i1[i, :num_selected_tokens].cpu().tolist())

    print(f'Problem size: ({num_tokens}, {hidden_dim})')
    print(pd.DataFrame({'latency': {
        'naive_routing': triton.testing.do_bench(lambda: naive_routing(inputs, gating_weight, num_experts, top_k)),
        'compiled_routing': triton.testing.do_bench(lambda: compiled_routing(inputs, gating_weight, num_experts, top_k)),
        'fused_routing': triton.testing.do_bench(lambda: fused_routing(inputs, gating_weight, num_experts, top_k)),
    }}))


if __name__ == "__main__":
    test_routing(num_tokens=4096, hidden_dim=4096, num_experts=128, top_k=8)
