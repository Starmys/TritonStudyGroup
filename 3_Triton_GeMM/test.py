import torch

from triton_gemm import triton_matmul


def profile(func, inputs, num_warmups=50, num_iters=50):
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
    c_triton = triton_matmul(a, b)

    torch.testing.assert_close(c_triton, c_torch, atol=1e-3, rtol=1e-3)

    latency_torch = profile(torch.matmul, (a, b))
    latency_triton = profile(triton_matmul, (a, b))

    print(f'Problem size: ({M}, {N}, {K})')
    print(f'Torch GeMM Latency: {latency_torch:.3f} ms')
    print(f'Triton GeMM Latency: {latency_triton:.3f} ms')


if __name__ == '__main__':
    test_matmul(M=4096, N=4096, K=4096)
