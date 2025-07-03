import torch

from my_softmax import naive_softmax


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
    y_naive = naive_softmax(x)

    torch.testing.assert_close(y_naive, y_torch)

    latency_torch = profile(torch_softmax, (x, ))
    latency_naive = profile(naive_softmax, (x, ))

    print(f'Batch size: {batch_size}, Hidden dim: {hidden_dim}')
    print(f'Torch Softmax Latency: {latency_torch:.2f} ms')
    print(f'Naive Softmax Latency: {latency_naive:.2f} ms')


if __name__ == '__main__':
    test_softmax(batch_size=4567, hidden_dim=4096)
