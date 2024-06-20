# Copyright JKU Linz 2023
# Author: Maximilian Beck
import math

import torch
import torch.nn.functional as F

"""In this file we implement the tiled version of the forward pass of the VLSTM model.

The tiled version is used for the kernel implementation of the model.
"""


def vlstm_fw_tiled_torch(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    bq_tile_size: int = -1,
    bkv_tile_size: int = -1,
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
            m_prev = m
            l_prev = l
            n_prev = n
        # print(m)
        h_matrix[:, :, q_idx * bq_tile_size : (q_idx + 1) * bq_tile_size, :] = h_tile

    return h_matrix, m, l, n


# non-tiled reference implementation
def vlstm_fw_torch_ref(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    eps: float = 0.0,
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
    # log_fgates = (
    #     fgate_preact  # (B, NH, S, 1) #! We do not apply sigmoid here for debugging
    # )

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
    # log_D_matrix = log_fg_matrix  # (B, NH, S, S)
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
    return retrieved_values, normalizer, max_log_D, l, D_matrix


def vlstm_parallel_fw_torch_w_groupnorm(
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
    normalizer = torch.maximum(
        C_matrix.sum(dim=-1, keepdim=True).abs(), torch.exp(-max_log_D)
    )  # (B, NH, S, 1)
    # (B, NH, S, S)
    C_matrix_normalized = C_matrix / (normalizer + eps)

    # hidden states
    hiddens = C_matrix_normalized @ values  # (B, NH, S, DH)

    return hiddens, normalizer, max_log_D


def vlstm_parallel_bw_torch_w_groupnorm(
    delta_Htilde: torch.Tensor,
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    var_n: torch.Tensor,
    var_m: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, ...]:
    B, NH, S, DH = queries.shape
    _dtype, _device = queries.dtype, queries.device

    # compute var_D
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
    ltr_ig = torch.where(ltr, 0.0, -float("inf"))
    ig_matrix = igate_preact.transpose(-2, -1) + ltr_ig  # (B, NH, S, S)
    var_Dtilde = log_fg_matrix + ig_matrix
    var_D = torch.exp(var_Dtilde - var_m)

    # intermediate delta-errors
    delta_C = delta_Htilde @ values.transpose(-2, -1) / (var_n + eps)

    var_QK = queries @ (keys / math.sqrt(DH)).transpose(-2, -1)

    delta_D = delta_C * var_QK

    delta_Dtilde = delta_D * var_D

    # compute fgate and igate preact delta errors
    # delta_f: forget gate preactivation delta errors
    ltr_dm1 = torch.tril(
        torch.ones(
            (S, S),
            dtype=torch.bool,
            device=_device,
        ),
        diagonal=-1,  #! Also mask out the diagonal as it is constant 1 in the D matrix
    )
    masked_deltaDtilde = torch.where(
        ltr_dm1,
        delta_Dtilde,
        torch.tensor(0.0, device=_device, dtype=_dtype),
    )

    delta_fbar = torch.zeros((B, NH, S, 1), device=_device, dtype=_dtype)
    # # first forget gate index (k=0) does not get a gradient (since it is not used in the forward pass)
    # TODO implement this in a more efficient way
    for k in range(1, S):
        for j in range(k):
            delta_fbar[:, :, k, 0] += (
                masked_deltaDtilde[:, :, k:, j].view(B, NH, -1).sum(dim=-1)
            )

    delta_f = delta_fbar * torch.sigmoid(-fgate_preact)

    # delta_i: input gate preactivation delta errors
    delta_i = torch.sum(delta_Dtilde, dim=-2).unsqueeze_(-1)

    # output delta-errors / gradients
    mat_P = delta_C * var_D
    delta_Q = mat_P @ (keys / math.sqrt(DH))
    delta_K = mat_P.transpose(-2, -1) @ (queries / math.sqrt(DH))

    var_C = var_QK * var_D
    delta_V = var_C.transpose(-2, -1) @ (delta_Htilde / (var_n + eps))
    return (
        delta_Q,
        delta_K,
        delta_V,
        delta_i,
        delta_f,
        delta_D,
        delta_Dtilde,
        delta_fbar,
        mat_P,
        var_C,
    )


def vlstm_parallel_fwbw_torch_w_groupnorm(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hiddens, var_n, var_m = vLSTMParallelFwBwWithGroupNorm.apply(
        queries, keys, values, igate_preact, fgate_preact, eps
    )
    return hiddens, var_n, var_m


class vLSTMParallelFwBwWithGroupNorm(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        igate_preact: torch.Tensor,
        fgate_preact: torch.Tensor,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hiddens, var_n, var_m = vlstm_parallel_fw_torch_w_groupnorm(
            queries=queries,
            keys=keys,
            values=values,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
            eps=eps,
        )
        ctx.save_for_backward(
            queries, keys, values, igate_preact, fgate_preact, var_n, var_m
        )
        return hiddens, var_n, var_m

    @staticmethod
    def backward(
        ctx,
        delta_Htilde: torch.Tensor,
        grad_var_n_unused: torch.Tensor,
        grad_var_m_unused: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        (queries, keys, values, igate_preact, fgate_preact, var_n, var_m) = (
            ctx.saved_tensors
        )
        delta_Q, delta_K, delta_V, delta_i, delta_f = (
            vlstm_parallel_bw_torch_w_groupnorm(
                delta_Htilde=delta_Htilde,
                queries=queries,
                keys=keys,
                values=values,
                igate_preact=igate_preact,
                fgate_preact=fgate_preact,
                var_n=var_n,
                var_m=var_m,
            )
        )
        return delta_Q, delta_K, delta_V, delta_i, delta_f, None
