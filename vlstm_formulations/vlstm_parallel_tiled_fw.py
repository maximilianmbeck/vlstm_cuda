# Copyright JKU Linz 2024
# Author: Maximilian Beck
import math

import torch
import torch.nn.functional as F

"""In this file we implement the tiled version of the forward pass of the VLSTM model.

The tiled version is used for the kernel implementation of the model.
"""


def construct_log_gate_matrix_paper(
    fgs: torch.Tensor, igs: torch.Tensor
) -> torch.Tensor:
    _device = fgs.device
    _dtype = fgs.dtype
    B, NH, S, _ = fgs.shape
    ltr = torch.tril(
        torch.ones(
            (S, S),
            dtype=torch.bool,
            device=_device,
        )
    )
    log_fgates_cumsum = torch.cat(
        [
            torch.zeros((B, NH, 1, 1), dtype=_dtype, device=_device),
            torch.cumsum(fgs, dim=-2),
        ],
        dim=-2,
    )  # (B, NH, S+1, 1)
    # for each batch/head this is a matrix of shape (S+1, S+1) containing the cumsum of the log forget gate values
    # in the second dimension (colum dimension). Each row has the same is a copy of the first row.
    # First entry of each row is zero.
    rep_log_fgates_cumsum = log_fgates_cumsum.repeat(
        1, 1, 1, S + 1
    )  # (B, NH, S+1, S+1)
    # Now in each row cut off / subtract the forgetgate values of the later timesteps
    # where col j > row i
    _log_fg_matrix = rep_log_fgates_cumsum - rep_log_fgates_cumsum.transpose(
        -2, -1
    )  # (B, NH, S+1, S+1)
    # Causal masking & selection of the correct submatrix, such that forgetgate at timestep t is not applied
    # to the input at timestep t
    log_fg_matrix = torch.where(
        ltr, _log_fg_matrix[:, :, 1:, 1:], -float("inf")
    )  # (B, NH, S, S)

    # gate decay matrix D (combination of forget gate and input gate)
    log_D_matrix = log_fg_matrix + igs.transpose(-2, -1)  # (B, NH, S, S)
    return log_D_matrix


def construct_log_gate_matrix_tiled(
    fgs: torch.Tensor,
    igs: torch.Tensor,
    BQ: int,
    BKV: int,
    idx_BQ: int,
    idx_BKV: int,
    fgs_cs: torch.Tensor = None,
) -> torch.Tensor:
    B, NH, S = fgs.shape
    if fgs_cs is None:
        fgs_cs = torch.cumsum(fgs, dim=-1)
    fgs_cs_chunk_Q = fgs_cs[:, :, idx_BQ * BQ : (idx_BQ + 1) * BQ]
    fgs_cs_chunk_KV = fgs_cs[:, :, idx_BKV * BKV : (idx_BKV + 1) * BKV]

    fgate_tile = fgs_cs_chunk_Q[:, :, :, None] - fgs_cs_chunk_KV[:, :, None, :]

    igs_chunk = igs[:, :, idx_BKV * BKV : (idx_BKV + 1) * BKV]
    log_D_matrix = fgate_tile + igs_chunk

    # causal masking
    if idx_BKV * BKV >= idx_BQ * BQ:
        bq_idxes = torch.arange(idx_BQ * BQ, (idx_BQ + 1) * BQ)
        kv_idxes = torch.arange(idx_BKV * BKV, (idx_BKV + 1) * BKV)
        idx_mask = (
            bq_idxes[:, None] - kv_idxes[None, :]
        )  # or bq_idxes[:, None] >= kv_idxes[None, :]
        log_D_matrix = torch.where(idx_mask < 0, -float("inf"), log_D_matrix)
    return log_D_matrix


def vlstm_parallel_tiled(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    bq_tile_size: int = -1,
    bkv_tile_size: int = -1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """This is the core vLSTM operation in parallel form computed in tiles.
    This version is stabilized. We control the range of exp() arguments by
    ensuring that they are always smaller than 0.0 by subtracting the maximum.

    Args:
        queries (torch.Tensor): (B, NH, S, DH)
        keys (torch.Tensor): (B, NH, S, DH)
        values (torch.Tensor): (B, NH, S, DH)
        igate_preact (torch.Tensor): (B, NH, S, 1)
        fgate_preact (torch.Tensor): (B, NH, S, 1)
        bq_tile_size (int, optional): Tile size along sequence dim for queries. Defaults to -1.
                                      If -1, no tiling is performed.
        bkv_tile_size (int, optional): Tile size along sequence dim for keys and values. Defaults to -1.
                                        If -1, no tiling is performed.

    Returns:
        torch.Tensor: (B, NH, S, DH), retrieved values
    """

    B, NH, S, DH = queries.shape
    _dtype, _device = queries.dtype, queries.device
    if bq_tile_size == -1:
        bq_tile_size = S
    else:
        assert S % bq_tile_size == 0, "S must be divisible by bq_tile_size"
    if bkv_tile_size == -1:
        bkv_tile_size = S
    else:
        assert S % bkv_tile_size == 0, "S must be divisible by bkv_tile_size"

    #! We compute the gate matrix D in non tiled way:
    # forget gate matrix
    log_fgates = F.logsigmoid(fgate_preact)  # (B, NH, S, 1)
    ltr = torch.tril(
        torch.ones(
            (S, S),
            dtype=torch.bool,
            device=_device,
        )
    )
    log_fgates_cumsum = torch.cat(
        [
            torch.zeros((B, NH, 1, 1), dtype=_dtype, device=_device),
            torch.cumsum(log_fgates, dim=-2),
        ],
        dim=-2,
    )  # (B, NH, S+1, 1)
    # for each batch/head this is a matrix of shape (S+1, S+1) containing the cumsum of the log forget gate values
    # in the second dimension (colum dimension). Each row has the same is a copy of the first row.
    # First entry of each row is zero.
    rep_log_fgates_cumsum = log_fgates_cumsum.repeat(
        1, 1, 1, S + 1
    )  # (B, NH, S+1, S+1)
    # Now in each row cut off / subtract the forgetgate values of the later timesteps
    # where col j > row i
    _log_fg_matrix = rep_log_fgates_cumsum - rep_log_fgates_cumsum.transpose(
        -2, -1
    )  # (B, NH, S+1, S+1)
    # Causal masking & selection of the correct submatrix, such that forgetgate at timestep t is not applied
    # to the input at timestep t
    log_fg_matrix = torch.where(
        ltr, _log_fg_matrix[:, :, 1:, 1:], -float("inf")
    )  # (B, NH, S, S)

    # gate decay matrix D (combination of forget gate and input gate)
    log_D_matrix = log_fg_matrix + igate_preact.transpose(-2, -1)  # (B, NH, S, S)

    #! From here begin tiling:
    q_tiles = torch.split(queries, bq_tile_size, dim=2)
    k_tiles = torch.split(keys, bkv_tile_size, dim=2)
    v_tiles = torch.split(values, bkv_tile_size, dim=2)
    print(f"q_tiles: {len(q_tiles)}, {q_tiles[0].shape}")
    print(f"kv_tiles: {len(k_tiles)}, {k_tiles[0].shape}")

    # we do not break causality since the log_fg_matrix is already causal

    h_matrix = torch.zeros_like(queries)  # the output matrix
    for q_idx, q_tile in enumerate(q_tiles):
        m_prev = torch.zeros((B, NH, bq_tile_size, 1), dtype=_dtype, device=_device)
        l_prev = torch.zeros((B, NH, bq_tile_size, 1), dtype=_dtype, device=_device)
        n_prev = torch.zeros((B, NH, bq_tile_size, 1), dtype=_dtype, device=_device)
        h_tile = torch.zeros_like(q_tile)
        for kv_idx, (k_tile, v_tile) in enumerate(zip(k_tiles, v_tiles)):
            # print(f"q_idx: {q_idx*bq_tile_size}, kv_idx: {kv_idx*bkv_tile_size}")
            d_tile = log_D_matrix[
                :,
                :,
                q_idx * bq_tile_size : (q_idx + 1) * bq_tile_size,
                kv_idx * bkv_tile_size : (kv_idx + 1) * bkv_tile_size,
            ]
            s_tile = q_tile @ (k_tile.transpose(-2, -1) / math.sqrt(DH))
            if kv_idx == 0:
                m, _ = torch.max(d_tile, dim=-1, keepdim=True)
                l = (s_tile * torch.exp(d_tile - m)).sum(dim=-1, keepdim=True)
            else:
                m = torch.maximum(m_prev, torch.max(d_tile, dim=-1, keepdim=True)[0])
                l = torch.exp(m_prev - m) * l_prev + (
                    s_tile * torch.exp(d_tile - m)
                ).sum(dim=-1, keepdim=True)

            n = torch.maximum(torch.abs(l), torch.exp(-m))
            c_tile = (s_tile * torch.exp(d_tile - m)) / (n + eps)

            h_tile = torch.exp(m_prev - m) * (n_prev / n) * h_tile + c_tile @ v_tile

            if q_idx == 10:
                print(
                    f"q_idx: {q_idx}, kv_idx: {kv_idx}, m_prev: {m_prev}, m: {m}, l_prev: {l_prev}, l: {l}, n_prev: {n_prev}, n: {n}"
                )
            m_prev = m
            l_prev = l
            n_prev = n
        h_matrix[:, :, q_idx * bq_tile_size : (q_idx + 1) * bq_tile_size, :] = h_tile

    return h_matrix, m, l, n


def vlstm_parallel_tiled_stable(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    bq_tile_size: int = -1,
    bkv_tile_size: int = -1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """This is the core vLSTM operation in parallel form computed in tiles.
    This version is stabilized. We control the range of exp() arguments by
    ensuring that they are always smaller than 0.0 by subtracting the maximum.

    Args:
        queries (torch.Tensor): (B, NH, S, DH)
        keys (torch.Tensor): (B, NH, S, DH)
        values (torch.Tensor): (B, NH, S, DH)
        igate_preact (torch.Tensor): (B, NH, S, 1)
        fgate_preact (torch.Tensor): (B, NH, S, 1)
        bq_tile_size (int, optional): Tile size along sequence dim for queries. Defaults to -1.
                                      If -1, no tiling is performed.
        bkv_tile_size (int, optional): Tile size along sequence dim for keys and values. Defaults to -1.
                                        If -1, no tiling is performed.

    Returns:
        torch.Tensor: (B, NH, S, DH), retrieved values
    """
    # TODO from here!
    # add second stabilizer state for exp(-m(2))

    B, NH, S, DH = queries.shape
    _dtype, _device = queries.dtype, queries.device
    if bq_tile_size == -1:
        bq_tile_size = S
    else:
        assert S % bq_tile_size == 0, "S must be divisible by bq_tile_size"
    if bkv_tile_size == -1:
        bkv_tile_size = S
    else:
        assert S % bkv_tile_size == 0, "S must be divisible by bkv_tile_size"

    #! We compute the gate matrix D in non tiled way:
    # forget gate matrix
    log_fgates = F.logsigmoid(fgate_preact)  # (B, NH, S, 1)
    ltr = torch.tril(
        torch.ones(
            (S, S),
            dtype=torch.bool,
            device=_device,
        )
    )
    log_fgates_cumsum = torch.cat(
        [
            torch.zeros((B, NH, 1, 1), dtype=_dtype, device=_device),
            torch.cumsum(log_fgates, dim=-2),
        ],
        dim=-2,
    )  # (B, NH, S+1, 1)
    # for each batch/head this is a matrix of shape (S+1, S+1) containing the cumsum of the log forget gate values
    # in the second dimension (colum dimension). Each row has the same is a copy of the first row.
    # First entry of each row is zero.
    rep_log_fgates_cumsum = log_fgates_cumsum.repeat(
        1, 1, 1, S + 1
    )  # (B, NH, S+1, S+1)
    # Now in each row cut off / subtract the forgetgate values of the later timesteps
    # where col j > row i
    _log_fg_matrix = rep_log_fgates_cumsum - rep_log_fgates_cumsum.transpose(
        -2, -1
    )  # (B, NH, S+1, S+1)
    # Causal masking & selection of the correct submatrix, such that forgetgate at timestep t is not applied
    # to the input at timestep t
    log_fg_matrix = torch.where(
        ltr, _log_fg_matrix[:, :, 1:, 1:], -float("inf")
    )  # (B, NH, S, S)

    # gate decay matrix D (combination of forget gate and input gate)
    log_D_matrix = log_fg_matrix + igate_preact.transpose(-2, -1)  # (B, NH, S, S)

    #! From here begin tiling:
    q_tiles = torch.split(queries, bq_tile_size, dim=2)
    k_tiles = torch.split(keys, bkv_tile_size, dim=2)
    v_tiles = torch.split(values, bkv_tile_size, dim=2)
    print(f"q_tiles: {len(q_tiles)}, {q_tiles[0].shape}")
    print(f"kv_tiles: {len(k_tiles)}, {k_tiles[0].shape}")

    # we do not break causality since the log_fg_matrix is already causal

    h_matrix = torch.zeros_like(queries)  # the output matrix
    for q_idx, q_tile in enumerate(q_tiles):
        m_prev = torch.zeros((B, NH, bq_tile_size, 1), dtype=_dtype, device=_device)
        l_prev = torch.zeros((B, NH, bq_tile_size, 1), dtype=_dtype, device=_device)
        n_prev = torch.zeros((B, NH, bq_tile_size, 1), dtype=_dtype, device=_device)
        h_tile = torch.zeros_like(q_tile)
        for kv_idx, (k_tile, v_tile) in enumerate(zip(k_tiles, v_tiles)):
            # print(f"q_idx: {q_idx*bq_tile_size}, kv_idx: {kv_idx*bkv_tile_size}")
            d_tile = log_D_matrix[
                :,
                :,
                q_idx * bq_tile_size : (q_idx + 1) * bq_tile_size,
                kv_idx * bkv_tile_size : (kv_idx + 1) * bkv_tile_size,
            ]
            s_tile = q_tile @ (k_tile.transpose(-2, -1) / math.sqrt(DH))

            m_temp = torch.maximum(
                torch.tensor([[[-10.0]]], dtype=_dtype, device=_device),
                torch.max(d_tile, dim=-1, keepdim=True)[0],
            )

            m = torch.maximum(m_prev, m_temp)
            l = torch.exp(m_prev - m) * l_prev + (s_tile * torch.exp(d_tile - m)).sum(
                dim=-1, keepdim=True
            )

            n = torch.maximum(torch.abs(l), torch.exp(-m))
            c_tile = (s_tile * torch.exp(d_tile - m)) / (n + eps)

            h_tile = torch.exp(m_prev - m) * (n_prev / n) * h_tile + c_tile @ v_tile

            if q_idx == 10:
                print(
                    f"q_idx: {q_idx}, kv_idx: {kv_idx}, m_prev: {m_prev}, m: {m}, l_prev: {l_prev}, l: {l}, n_prev: {n_prev}, n: {n}"
                )
            m_prev = m
            l_prev = l
            n_prev = n
        h_matrix[:, :, q_idx * bq_tile_size : (q_idx + 1) * bq_tile_size, :] = h_tile

    return h_matrix, m, l, n


# non-tiled reference implementation
def vlstm_parallel_fw_torch_ref(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """This is the core vLSTM operation in parallel form.
    This version is stabilized. We control the range of exp() arguments by
    ensuring that they are always smaller than 0.0 by subtracting the maximum.

    Args:
        queries (torch.Tensor): (B, NH, S, DH)
        keys (torch.Tensor): (B, NH, S, DH)
        values (torch.Tensor): (B, NH, S, DH)
        igate_preact (torch.Tensor): (B, NH, S, 1)
        fgate_preact (torch.Tensor): (B, NH, S, 1)
        stabilize_rowwise (bool, optional): Wether to stabilize the combination matrix C rowwise (take maximum per row).
            Alternative: Subtract the maximum over all rows. Defaults to True.

    Returns:
        torch.Tensor: (B, NH, S, DH), retrieved values
    """

    B, NH, S, DH = queries.shape
    _dtype, _device = queries.dtype, queries.device

    # forget gate matrix
    log_fgates = F.logsigmoid(fgate_preact)  # (B, NH, S, 1)
    ltr = torch.tril(
        torch.ones(
            (S, S),
            dtype=torch.bool,
            device=_device,
        )
    )
    log_fgates_cumsum = torch.cat(
        [
            torch.zeros((B, NH, 1, 1), dtype=_dtype, device=_device),
            torch.cumsum(log_fgates, dim=-2),
        ],
        dim=-2,
    )  # (B, NH, S+1, 1)
    # for each batch/head this is a matrix of shape (S+1, S+1) containing the cumsum of the log forget gate values
    # in the second dimension (colum dimension). Each row has the same is a copy of the first row.
    # First entry of each row is zero.
    rep_log_fgates_cumsum = log_fgates_cumsum.repeat(
        1, 1, 1, S + 1
    )  # (B, NH, S+1, S+1)
    # Now in each row cut off / subtract the forgetgate values of the later timesteps
    # where col j > row i
    _log_fg_matrix = rep_log_fgates_cumsum - rep_log_fgates_cumsum.transpose(
        -2, -1
    )  # (B, NH, S+1, S+1)
    # Causal masking & selection of the correct submatrix, such that forgetgate at timestep t is not applied
    # to the input at timestep t
    log_fg_matrix = torch.where(
        ltr, _log_fg_matrix[:, :, 1:, 1:], -float("inf")
    )  # (B, NH, S, S)

    # gate decay matrix D (combination of forget gate and input gate)
    log_D_matrix = log_fg_matrix + igate_preact.transpose(-2, -1)  # (B, NH, S, S)
    # D matrix stabilization
    max_log_D, _ = torch.max(log_D_matrix, dim=-1, keepdim=True)  # (B, NH, S, 1)
    log_D_matrix_stabilized = log_D_matrix - max_log_D  # (B, NH, S, S)
    D_matrix = torch.exp(log_D_matrix_stabilized)  # (B, NH, S, S)

    keys_scaled = keys / math.sqrt(DH)

    # combination matrix C
    qk_matrix = queries @ keys_scaled.transpose(-2, -1)  # (B, NH, S, S)
    C_matrix = qk_matrix * D_matrix  # (B, NH, S, S)
    l = C_matrix.sum(dim=-1, keepdim=True)
    normalizer = torch.maximum(l.abs(), torch.exp(-max_log_D))  # (B, NH, S, 1)
    # (B, NH, S, S)
    C_matrix_normalized = C_matrix / (normalizer + eps)

    # retrieved values
    retrieved_values = C_matrix_normalized @ values  # (B, NH, S, DH)
    return retrieved_values, log_D_matrix, C_matrix_normalized
