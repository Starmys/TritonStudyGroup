import torch

from triton_naive_gemm import triton_matmul as triton_naive_matmul
from triton_good_gemm import triton_matmul as triton_good_matmul
from triton_better_gemm import triton_matmul as triton_better_matmul


def call_matmul(M: int, N: int, K: int):
    a = torch.randn((M, K), dtype=torch.float16, requires_grad=False).cuda()
    b = torch.randn((K, N), dtype=torch.float16, requires_grad=False).cuda()

    torch.matmul(a, b)
    triton_naive_matmul(a, b)
    triton_good_matmul(a, b)
    triton_better_matmul(a, b)


if __name__ == '__main__':
    call_matmul(M=4096, N=4096, K=4096)
    call_matmul(M=4321, N=4080, K=4080)
