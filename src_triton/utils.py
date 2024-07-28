import torch


def lower_triangular_block_matrix(n, block_size, device, dtype):
    assert n % block_size == 0
    n_blocks = n // block_size

    mat = torch.zeros((n, n), device=device, dtype=dtype)
    for i in range(n_blocks):
        mat[i * block_size : (i + 1) * block_size, : (i + 1) * block_size] = 1.0
    return mat
