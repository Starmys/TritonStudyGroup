import torch

from triton_naive_gemm import triton_matmul as triton_naive_matmul
from triton_good_gemm import triton_matmul as triton_good_matmul
from triton_better_gemm import triton_matmul as triton_better_matmul


def profile(func, inputs, num_warmups=200, num_iters=200):
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


def test_matmul(M: int, N: int, K: int):
    a = torch.randn((M, K), dtype=torch.float16, device='cuda', requires_grad=False)
    b = torch.randn((K, N), dtype=torch.float16, device='cuda', requires_grad=False)

    c_torch = torch.matmul(a, b)
    c_triton_naive = triton_naive_matmul(a, b)
    c_triton_good = triton_good_matmul(a, b)
    c_triton_better = triton_better_matmul(a, b)

    torch.testing.assert_close(c_triton_naive, c_torch, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(c_triton_good, c_torch, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(c_triton_better, c_torch, atol=1e-3, rtol=1e-3)

    latency_torch = profile(torch.matmul, (a, b))
    latency_triton_naive = profile(triton_naive_matmul, (a, b))
    latency_triton_good = profile(triton_good_matmul, (a, b))
    latency_triton_better = profile(triton_better_matmul, (a, b))

    print(f'Problem size: ({M}, {N}, {K})')
    print(f'Torch GeMM Latency: {latency_torch:.3f} ms')
    print(f'Triton (Naive) GeMM Latency: {latency_triton_naive:.3f} ms')
    print(f'Triton (Good) GeMM Latency: {latency_triton_good:.3f} ms')
    print(f'Triton (Better) GeMM Latency: {latency_triton_better:.3f} ms')


if __name__ == '__main__':
    test_matmul(M=4096, N=4096, K=4096)
    test_matmul(M=4321, N=4080, K=4080)
