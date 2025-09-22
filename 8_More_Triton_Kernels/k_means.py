import numpy as np
import pandas as pd
import torch
import triton
import triton.language as tl


@triton.jit
def _triton_assign_kernel(
    K, X, S, C, M,  # data, centroids, data_sum, data_cnt, max_idx
    stride_kz, stride_kn, stride_kd,
    stride_xz, stride_xk, stride_xd,
    stride_sz, stride_sk, stride_sd,
    stride_cz, stride_ck,
    stride_mz, stride_mn,
    num_tokens, num_centroids,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_n = tl.program_id(0) * BLOCK_N
    batch_idx = tl.program_id(1)

    if start_n >= num_tokens:
        return

    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    n_mask = offs_n < num_tokens

    k_ptrs = K + batch_idx * stride_kz + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    x_ptrs = X + batch_idx * stride_xz + offs_k[None, :] * stride_xk + offs_d[:, None] * stride_xd
    s_ptrs = S + batch_idx * stride_sz + offs_d[None, :] * stride_sd
    c_ptrs = C + batch_idx * stride_cz
    m_ptrs = M + batch_idx * stride_mz + offs_n * stride_mn

    k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.)
    max_val = tl.zeros([BLOCK_N], dtype=tl.float32) - float("inf")
    max_idx = tl.zeros([BLOCK_N], dtype=tl.int32)

    for start_k in range(0, num_centroids, BLOCK_K):
        x = tl.load(x_ptrs)
        ip = tl.dot(k, x)
        tmp_max_val, tmp_max_idx = tl.max(ip, axis=1, return_indices=True)
        tmp_max_idx += start_k
        max_idx = tl.where(tmp_max_val > max_val, tmp_max_idx, max_idx)
        max_val = tl.maximum(tmp_max_val, max_val)
        x_ptrs += BLOCK_K * stride_xk

    tl.store(m_ptrs, max_idx, mask=n_mask)
    tl.atomic_add(s_ptrs + max_idx[:, None] * stride_sk, k.to(S.type.element_ty), mask=n_mask[:, None], sem='relaxed')
    tl.atomic_add(c_ptrs + max_idx * stride_ck, tl.zeros_like(max_idx) + 1, mask=n_mask, sem='relaxed')


@triton.jit
def _triton_update_kernel(
    X, S, C,  # centroids, data_sum, data_cnt
    stride_xz, stride_xk, stride_xd,
    stride_sz, stride_sk, stride_sd,
    stride_cz, stride_ck,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
    NORMORLIZE: tl.constexpr,
):
    start_k = tl.program_id(0) * BLOCK_K
    batch_idx = tl.program_id(1)

    offs_k = start_k + tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    x_ptrs = X + batch_idx * stride_xz + offs_k[:, None] * stride_xk + offs_d[None, :] * stride_xd
    s_ptrs = S + batch_idx * stride_sz + offs_k[:, None] * stride_sk + offs_d[None, :] * stride_sd
    c_ptrs = C + batch_idx * stride_cz + offs_k[:, None] * stride_ck

    s = tl.load(s_ptrs)
    c = tl.load(c_ptrs)
    x_mask = c > 0
    x = s / c
    if NORMORLIZE:
        x /= tl.sqrt(tl.sum(x * x, axis=-1, keep_dims=True))

    tl.store(x_ptrs, x.to(X.type.element_ty), mask=x_mask)


def _triton_k_means_train(
    data: torch.Tensor,       # [batch_size, num_tokens, dim]
    centroids: torch.Tensor,  # [batch_size, num_centroids, dim]
    normalize_centroids: bool = True,
    return_indices: bool = False,
):
    batch_size, num_tokens, dim = data.shape
    num_centroids = centroids.shape[1]
    max_idx = torch.empty((batch_size, num_tokens), dtype=torch.int32, device=data.device)
    data_sum = torch.zeros_like(centroids, dtype=torch.float32)
    data_cnt = torch.zeros((batch_size, num_centroids), dtype=torch.int32, device=data.device)
    block_N = 128
    block_K = 64
    assert num_centroids % block_K == 0
    assert dim in [32, 64, 128]
    _triton_assign_kernel[(triton.cdiv(num_tokens, block_N), batch_size, 1)](
        data, centroids, data_sum, data_cnt, max_idx,
        data.stride(0), data.stride(1), data.stride(2),
        centroids.stride(0), centroids.stride(1), centroids.stride(2),
        data_sum.stride(0), data_sum.stride(1), data_sum.stride(2),
        data_cnt.stride(0), data_cnt.stride(1),
        max_idx.stride(0), max_idx.stride(1),
        num_tokens, num_centroids,
        BLOCK_N=block_N, BLOCK_K=block_K, BLOCK_D=dim,
        num_warps=4, num_stages=2,
    )
    block_K = 128
    _triton_update_kernel[(triton.cdiv(num_centroids, block_K), batch_size, 1)](
        centroids, data_sum, data_cnt,
        centroids.stride(0), centroids.stride(1), centroids.stride(2),
        data_sum.stride(0), data_sum.stride(1), data_sum.stride(2),
        data_cnt.stride(0), data_cnt.stride(1),
        BLOCK_K=block_K, BLOCK_D=dim,
        NORMORLIZE=normalize_centroids,
        num_warps=4, num_stages=1,
    )
    if return_indices:
        return centroids, max_idx, data_cnt.max().item()
    return centroids


def _torch_k_means_train(
    data: torch.Tensor,       # [batch_size, num_tokens, dim]
    centroids: torch.Tensor,  # [batch_size, num_centroids, dim]
    normalize_centroids: bool = True,
    return_indices: bool = False,
):
    batch_size, num_tokens, dim = data.shape
    num_centroids = centroids.shape[1]

    inner_prod = torch.einsum('bnd, bkd -> bnk', data, centroids)
    max_idx = torch.argmax(inner_prod, dim=-1)

    data_sum = []
    data_cnt = []
    ones_centroids = torch.ones((num_centroids, ), dtype=torch.int32, device=data.device)
    ones_tokens = torch.ones((num_tokens, ), dtype=torch.int32, device=data.device)
    for b in range(batch_size):
        data_sum.append(torch.index_add(input=centroids[b], dim=0, index=max_idx[b], source=data[b]))
        data_cnt.append(torch.index_add(input=ones_centroids, dim=0, index=max_idx[b], source=ones_tokens))
    data_cnt = torch.stack(data_cnt)
    new_centroids = torch.stack(data_sum) / data_cnt.unsqueeze(-1)

    if normalize_centroids:
        new_centroids /= new_centroids.norm(dim=-1, keepdim=True)
    new_centroids = new_centroids.to(centroids.dtype)

    if return_indices:
        return new_centroids, max_idx, data_cnt.max().item()
    return new_centroids


@triton.jit
def _triton_reverse_index_kernel(
    M, I, C,  # max_idx, clusters, cluster_size
    stride_mz, stride_mn,
    stride_iz, stride_ik, stride_in,
    stride_cz, stride_ck,
    num_tokens,
    BLOCK_N: tl.constexpr,
):
    start_n = tl.program_id(0) * BLOCK_N
    batch_idx = tl.program_id(1)

    if start_n >= num_tokens:
        return
    
    offs_n = start_n + tl.arange(0, BLOCK_N)
    n_mask = offs_n < num_tokens

    m_ptrs = M + batch_idx * stride_mz + offs_n * stride_mn
    i_ptrs = I + batch_idx * stride_iz
    c_ptrs = C + batch_idx * stride_cz

    max_idx = tl.load(m_ptrs, mask=n_mask, other=0)
    cnt = tl.atomic_add(c_ptrs + max_idx * stride_ck, tl.zeros_like(max_idx) + 1, mask=n_mask, sem='relaxed')
    tl.store(i_ptrs + max_idx * stride_ik + cnt * stride_in, offs_n, mask=n_mask)


def triton_reverse_index(
    max_idx: torch.Tensor,  # [batch_size, num_tokens]
    num_centroids: int,
    max_cluster_size: int,
):
    batch_size, num_tokens = max_idx.shape
    clusters = torch.zeros((batch_size, num_centroids, max_cluster_size), dtype=torch.int32, device=max_idx.device)
    cluster_size = torch.zeros((batch_size, num_centroids), dtype=torch.int32, device=max_idx.device)
    block_N = 128
    _triton_reverse_index_kernel[(triton.cdiv(num_tokens, block_N), batch_size, 1)](
        max_idx, clusters, cluster_size,
        max_idx.stride(0), 
        max_idx.stride(1),
        clusters.stride(0), clusters.stride(1), clusters.stride(2),
        cluster_size.stride(0), cluster_size.stride(1),
        num_tokens, BLOCK_N=block_N,
        num_warps=4, num_stages=1,
    )
    return clusters, cluster_size


def k_means(
    data: torch.Tensor,    # [batch_size, num_heads, num_tokens, head_dim]
    num_centroids: int,
    num_iters: int = 10,
    centroids_initialization: str = 'uniform',
    train_func = _triton_k_means_train,
):
    batch_size, num_heads, num_tokens, head_dim = data.shape

    if centroids_initialization == 'uniform':
        centroid_indices = torch.arange(num_centroids, dtype=torch.float32, device=data.device) * (num_tokens / num_centroids)
        centroid_indices += num_tokens / num_centroids / 2
        centroid_indices = centroid_indices.to(torch.int64)
    elif centroids_initialization == 'random':
        centroid_indices = torch.randperm(num_tokens, dtype=torch.int64, device=data.device)[:num_centroids]
    else:
        raise ValueError(f'unknown centroid initialization method: "{centroids_initialization}"')
    centroids = torch.index_select(data, dim=2, index=centroid_indices)

    data = data.reshape((-1, num_tokens, head_dim))
    centroids = centroids.reshape((-1, num_centroids, head_dim))
    for _ in range(num_iters - 1):
        centroids = train_func(data, centroids, normalize_centroids=True, return_indices=False)
    centroids, max_idx, max_cluster_size = train_func(data, centroids, normalize_centroids=False, return_indices=True)

    clusters, cluster_size = triton_reverse_index(max_idx, num_centroids, max_cluster_size)

    return centroids, clusters, cluster_size


def set_seed(seed):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def profile(func, inputs, num_warmups=1, num_iters=2):
    torch.cuda.synchronize()
    for _ in range(num_warmups):
        func(**inputs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        func(**inputs)
    end.record()
    torch.cuda.synchronize()
    latency = start.elapsed_time(end) / num_iters
    return latency


def compute_recall(ground_truth, I, retrieval_k=100):
    recall_list = []
    for i in range(len(I)):
        inter_sect = set(I[i][:retrieval_k]) & set(ground_truth[i][:retrieval_k])
        recall_list.append(len(inter_sect) / retrieval_k)
    recall_list = np.array(recall_list, dtype=np.float32)
    # print(recall_list)
    recall = np.mean(recall_list)
    return recall


def main(
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 128,
    num_tokens: int = 131072,
    num_centroids: int = 8192,
    num_iters: int = 10,
    random_seed: int = 2025,
    dtype: torch.dtype = torch.float16,
    device: torch.device = 'cuda',
):
    set_seed(random_seed)
    data = torch.randn((batch_size, num_heads, num_tokens, head_dim), dtype=dtype, device=device)
    print(f'Shape={data.shape}')

    inputs = {
        'data': data,
        'num_centroids': num_centroids,
        'num_iters': num_iters,
        'centroids_initialization': 'uniform',
    }

    print(pd.DataFrame({'latency': {
        'torch-kmeans': profile(k_means, {**inputs, 'train_func': _torch_k_means_train}),
        'triton-kmeans': profile(k_means, {**inputs, 'train_func': _triton_k_means_train}),
    }}).round(2))


if __name__ == '__main__':
    main(
        batch_size=1,
        num_heads=8,
        head_dim=128,
        num_tokens=131072,
        num_centroids=8192,
        num_iters=10,
        dtype=torch.float16,
    )
