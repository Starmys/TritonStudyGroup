import torch
from my_softmax.cuda import naive_softmax as naive_softmax_cuda, better_softmax as better_softmax_cuda


def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    return naive_softmax_cuda(x)


def better_softmax(x: torch.Tensor) -> torch.Tensor:
    return better_softmax_cuda(x)
