import torch
import triton
import pandas as pd

from flash_attn import flash_attn_func
from triton_flash_attn import triton_flash_attn_func, triton_flash_attn_split_bwd_func


def torch_naive_attn(
    q: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    softmax_scale: float = None,
    causal: bool = True,
):
    batch_size, num_tokens, num_heads, head_dim = q.shape
    softmax_scale = softmax_scale or head_dim ** (-0.5)
    s = torch.einsum('bmhd, bnhd -> bhmn', q, k).to(torch.float32) * softmax_scale
    if causal:
        arange = torch.arange(num_tokens, device=q.device)
        mask = arange[None, None, :, None] < arange[None, None, None, :]
        s.masked_fill_(mask, float('-inf'))
    p = torch.softmax(s, dim=-1).to(q.dtype)
    o = torch.einsum('bhmn, bnhd -> bmhd', p, v)
    return o


def test_attn(B: int, N: int, H: int, D: int, causal: bool):
    torch.manual_seed(4321)
    q = torch.empty((B, N, H, D), dtype=torch.bfloat16, device='cuda').normal_(mean=0.0, std=0.5).requires_grad_()
    k = torch.empty((B, N, H, D), dtype=torch.bfloat16, device='cuda').normal_(mean=0.0, std=0.5).requires_grad_()
    v = torch.empty((B, N, H, D), dtype=torch.bfloat16, device='cuda').normal_(mean=0.0, std=0.5).requires_grad_()
    do = torch.empty((B, N, H, D), dtype=torch.bfloat16, device='cuda').normal_(mean=0.0, std=0.5)
    softmax_scale = D ** (-0.5)
    inputs = {'q': q, 'k': k, 'v': v, 'softmax_scale': softmax_scale, 'causal': causal}

    def call_attn_fwd_bwd(attn_func):
        out = attn_func(**inputs)
        out.backward(do)
        dq, dk, dv = q.grad, k.grad, v.grad
        q.grad, k.grad, v.grad = None, None, None
        return out, dq, dk, dv

    # For auto-tune
    call_attn_fwd_bwd(triton_flash_attn_func)
    call_attn_fwd_bwd(triton_flash_attn_split_bwd_func)

    o_triton_0, dq_triton_0, dk_triton_0, dv_triton_0 = call_attn_fwd_bwd(triton_flash_attn_func)
    o_triton_1, dq_triton_1, dk_triton_1, dv_triton_1 = call_attn_fwd_bwd(triton_flash_attn_split_bwd_func)
    o_flash, dq_flash, dk_flash, dv_flash = call_attn_fwd_bwd(flash_attn_func)
    o_torch, dq_torch, dk_torch, dv_torch = call_attn_fwd_bwd(torch_naive_attn)

    atol = 1e-2 if causal else 5e-4
    rtol = 1e-3
    torch.testing.assert_close(o_flash, o_torch, atol=atol, rtol=rtol)
    torch.testing.assert_close(dq_flash, dq_torch, atol=atol, rtol=rtol)
    torch.testing.assert_close(dk_flash, dk_torch, atol=atol, rtol=rtol)
    torch.testing.assert_close(dv_flash, dv_torch, atol=atol, rtol=rtol)
    torch.testing.assert_close(o_triton_0, o_torch, atol=atol, rtol=rtol)
    torch.testing.assert_close(dq_triton_0, dq_torch, atol=atol, rtol=rtol)
    torch.testing.assert_close(dk_triton_0, dk_torch, atol=atol, rtol=rtol)
    torch.testing.assert_close(dv_triton_0, dv_torch, atol=atol, rtol=rtol)
    torch.testing.assert_close(o_triton_1, o_torch, atol=atol, rtol=rtol)
    torch.testing.assert_close(dq_triton_1, dq_torch, atol=atol, rtol=rtol)
    torch.testing.assert_close(dk_triton_1, dk_torch, atol=atol, rtol=rtol)
    torch.testing.assert_close(dv_triton_1, dv_torch, atol=atol, rtol=rtol)

    def profile_attn_fwd_bwd(attn_func):
        latency_fwd = triton.testing.do_bench(lambda: attn_func(**inputs))
        latency_bwd = triton.testing.do_bench(lambda: call_attn_fwd_bwd(attn_func)) - latency_fwd
        return {'fwd': latency_fwd, 'bwd': latency_bwd}

    print(f'Problem size: ({B}, {N}, {H}, {D}); Causal = {causal}')
    print(pd.DataFrame({
        'triton_atomic_add': profile_attn_fwd_bwd(triton_flash_attn_func),
        'triton_split_bwd': profile_attn_fwd_bwd(triton_flash_attn_split_bwd_func),
        'flash_official': profile_attn_fwd_bwd(flash_attn_func),
        'torch_naive': profile_attn_fwd_bwd(torch_naive_attn),
    }).T)


if __name__ == '__main__':
    print('='* 64)
    test_attn(B=1, H=32, N=4096, D=128, causal=False)
    print('-'* 64)
    test_attn(B=1, H=32, N=4096, D=128, causal=True)
    print('-'* 64)
    test_attn(B=1, H=32, N=4321, D=128, causal=False)
    print('-'* 64)
    test_attn(B=1, H=32, N=4321, D=128, causal=True)
    print('='* 64)
