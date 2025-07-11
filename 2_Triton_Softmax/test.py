import torch

from my_softmax import better_softmax as cuda_softmax
from triton_softmax import triton_softmax


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


def torch_softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softmax(x, dim=-1)


def test_softmax(batch_size: int, hidden_dim: int):
    x = torch.randn((batch_size, hidden_dim), device='cuda', requires_grad=False)

    y_torch = torch_softmax(x)
    y_cuda = cuda_softmax(x)
    y_triton = triton_softmax(x)

    torch.testing.assert_close(y_cuda, y_torch)
    torch.testing.assert_close(y_triton, y_torch)

    latency_torch = profile(torch_softmax, (x, ))
    latency_cuda = profile(cuda_softmax, (x, ))
    latency_triton = profile(triton_softmax, (x, ))

    print(f'Batch size: {batch_size}, Hidden dim: {hidden_dim}')
    print(f'Torch Softmax Latency: {latency_torch:.3f} ms')
    print(f'CUDA Softmax Latency: {latency_cuda:.3f} ms')
    print(f'Triton Softmax Latency: {latency_triton:.3f} ms')


if __name__ == '__main__':
    test_softmax(batch_size=8765, hidden_dim=4096)
