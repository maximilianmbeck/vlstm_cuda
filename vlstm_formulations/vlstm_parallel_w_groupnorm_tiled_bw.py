import math

import torch
import torch.nn.functional as F


def construct_log_gate_matrix_tiled(
    vecI: torch.Tensor,
    vecF: torch.Tensor,
    BQ: int,
    BKV: int,
    idx_BQ: int,
    idx_BKV: int,
    vecF_cs: torch.Tensor = None,
) -> torch.Tensor:
    B, NH, S = vecF.shape
    if vecF_cs is None:
        vecF_cs = torch.cumsum(vecF, dim=-1)
    vecF_cs_chunk_Q = vecF_cs[:, :, idx_BQ * BQ : (idx_BQ + 1) * BQ]
    vecF_cs_chunk_KV = vecF_cs[:, :, idx_BKV * BKV : (idx_BKV + 1) * BKV]

    vecF_cs_tile = vecF_cs_chunk_Q[:, :, :, None] - vecF_cs_chunk_KV[:, :, None, :]

    vecI_chunk = vecI[:, :, idx_BKV * BKV : (idx_BKV + 1) * BKV]
    matLogD_tile = vecF_cs_tile + vecI_chunk

    # causal masking
    if idx_BKV * BKV >= idx_BQ * BQ:
        bq_idxes = torch.arange(idx_BQ * BQ, (idx_BQ + 1) * BQ, device=vecI.device)
        kv_idxes = torch.arange(idx_BKV * BKV, (idx_BKV + 1) * BKV, device=vecI.device)
        idx_mask = (
            bq_idxes[:, None] - kv_idxes[None, :]
        )  # or bq_idxes[:, None] >= kv_idxes[None, :]
        matLogD_tile = torch.where(idx_mask < 0, -float("inf"), matLogD_tile)
    return matLogD_tile


def construct_log_gate_matrix_paper(
    vecI: torch.Tensor, vecF: torch.Tensor
) -> torch.Tensor:
    _device = vecF.device
    _dtype = vecF.dtype
    B, NH, S, _ = vecF.shape
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
            torch.cumsum(vecF, dim=-2),
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
    log_D_matrix = log_fg_matrix + vecI.transpose(-2, -1)  # (B, NH, S, S)
    return log_D_matrix


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def mlstm_parallel_w_groupnorm_torch_tiled_bw(
    matDeltaHtilde: torch.Tensor,
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
    vecI: torch.Tensor,
    vecF: torch.Tensor,
    vecM: torch.Tensor,
    vecN: torch.Tensor,
    BLOCK_Q: int = 32,
    BLOCK_KV: int = 32,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, ...]:

    B, NH, S, DH = matQ.shape
    _dtype, _device = matQ.dtype, matQ.device
    assert BLOCK_KV <= BLOCK_Q

    assert vecF.shape == (B, NH, S)
    assert vecI.shape == (B, NH, S)

    # ? preprocessing
    # precompute gate cumsums
    vecF_act = F.logsigmoid(vecF)
    vecF_cs = torch.cumsum(vecF_act, dim=-1)

    # ? tile the input tensors
    # we keep the batch and num_head dimensions
    # in a kernel we would embarrassingly parallelize over these dimensions

    # split along BLOCK_Q dimension:
    matQ_tiles = torch.split(matQ, BLOCK_Q, dim=2)
    matDeltaHtilde_tiles = torch.split(matDeltaHtilde, BLOCK_Q, dim=2)
    vecM_chunks = torch.split(vecM, BLOCK_Q, dim=2)
    vecN_chunks = torch.split(vecN, BLOCK_Q, dim=2)
    vecF_cs_chunks = torch.split(vecF_cs, BLOCK_Q, dim=2)

    # split along BLOCK_KV dimension:
    matK_tiles = torch.split(matK, BLOCK_KV, dim=2)
    matV_tiles = torch.split(matV, BLOCK_KV, dim=2)
    vecI_chunks = torch.split(vecI, BLOCK_KV, dim=2)

    # ? define the output tensors
    matDeltaQ = torch.zeros_like(matQ)
    matDeltaK = torch.zeros_like(matK)
    matDeltaV = torch.zeros_like(matV)
    vecDeltaI = torch.zeros_like(vecI)
    vecDeltaF = torch.zeros_like(vecF)

    print(
        f"matQ_tiles: {len(matQ_tiles)}, {matQ_tiles[0].shape} | matK_tiles: {len(matK_tiles)}, {matK_tiles[0].shape}"
    )

    # ? begin the backward pass

    #! KV dim loop
    # we will parallelize over this loop later
    # we start at the leftmost block of the KV dimension and work our way right
    for kvIdx, (matK_tile, matV_tile, vecI_chunk) in enumerate(
        zip(matK_tiles, matV_tiles, vecI_chunks)
    ):
        # init matDeltaK_tile, matDeltaV_tile to zero
        matDeltaK_tile = torch.zeros_like(matK_tile)
        matDeltaV_tile = torch.zeros_like(matV_tile)

        vecF_cs_chunk_KV = vecF_cs[:, :, kvIdx * BLOCK_KV : (kvIdx + 1) * BLOCK_KV]

        #! Q dim loop
        # we start at the diagonal of the S & D matrices and work our way down
        qStartIdx = (kvIdx * BLOCK_KV) // BLOCK_Q
        qEndIdx = ceildiv(S, BLOCK_Q)
        for qIdx in range(qStartIdx, qEndIdx):
            matQ_tile = matQ_tiles[qIdx]
            matDeltaHtilde_tile = matDeltaHtilde_tiles[qIdx]
            vecM_chunk = vecM_chunks[qIdx]
            vecN_chunk = vecN_chunks[qIdx]
            vecF_cs_chunk_Q = vecF_cs_chunks[qIdx]

            matDeltaC = (
                matDeltaHtilde_tile @ matV_tile.transpose(-2, -1) / (vecN_chunk + eps)
            )

            # ? recomputation of S & D matrices
            matS = (matQ_tile @ matK_tile.transpose(-2, -1)) / math.sqrt(DH)

            # construct D matrix
            vecF_cs_tile = (
                vecF_cs_chunk_Q[:, :, :, None] - vecF_cs_chunk_KV[:, :, None, :]
            )
            matLogD_tile = vecF_cs_tile + vecI_chunk
            # causal masking of matLogD_tile
            if kvIdx * BLOCK_KV >= qIdx * BLOCK_Q:
                bq_idxes = torch.arange(
                    qIdx * BLOCK_Q, (qIdx + 1) * BLOCK_Q, device=vecI.device
                )
                kv_idxes = torch.arange(
                    kvIdx * BLOCK_KV, (kvIdx + 1) * BLOCK_KV, device=vecI.device
                )
                idx_mask = bq_idxes[:, None] - kv_idxes[None, :]
                matLogD_tile = torch.where(idx_mask < 0, -float("inf"), matLogD_tile)

            matDtilde = torch.exp(matLogD_tile - vecM_chunk)
            # ? end recomputation of S & D matrices

            matDeltaCtilde = matDeltaC * matS * matDtilde

            # ? compute sums for vecDeltaF and vecDeltaI
            # TODO

            matP = matDeltaC * matDtilde
            matR = matS * matDtilde

            matDeltaQ_tile = matP @ (matK_tile / math.sqrt(DH))
            # * store matDeltaQ in HBM (this access is in parallel at the same HBM location, e.g. must be atomic)
            matDeltaQ[:, :, qIdx * BLOCK_Q : (qIdx + 1) * BLOCK_Q] += matDeltaQ_tile

            matDeltaK_tile += (matP.transpose(-2, -1) @ matQ_tile) / math.sqrt(DH)

            matDeltaV_tile += matR.transpose(-2, -1) @ (
                matDeltaHtilde_tile / (vecN_chunk + eps)
            )
            #! end Q dim loop

        # * store matDeltaK_tile & matDeltaV_tile in HBM (every thread block writes to a different HBM location)
        matDeltaK[:, :, kvIdx * BLOCK_KV : (kvIdx + 1) * BLOCK_KV] = matDeltaK_tile
        matDeltaV[:, :, kvIdx * BLOCK_KV : (kvIdx + 1) * BLOCK_KV] = matDeltaV_tile
        #! end KV dim loop

    return matDeltaQ, matDeltaK, matDeltaV, vecDeltaI, vecDeltaF, matDtilde


#! Reference for tiled version.
# Copy from /home/max/myrepos/vlstm_cuda/vlstm_formulations/vlstm_parallel_w_groupnorm.py with adapted naming.
def vlstm_parallel_w_groupnorm_torch_bw(
    matDeltaHtilde: torch.Tensor,
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
    vecI: torch.Tensor,
    vecF: torch.Tensor,
    vecM: torch.Tensor,
    vecN: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, ...]:
    B, NH, S, DH = matQ.shape
    _dtype, _device = matQ.dtype, matQ.device

    # compute var_D
    # forget gate matrix
    log_fgates = F.logsigmoid(vecF)  # (B, NH, S, 1)
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
    ig_matrix = vecI.transpose(-2, -1) + ltr_ig  # (B, NH, S, S)
    var_Dtilde = log_fg_matrix + ig_matrix
    var_D = torch.exp(var_Dtilde - vecM)

    # intermediate delta-errors
    delta_C = matDeltaHtilde @ matV.transpose(-2, -1) / (vecN + eps)

    var_QK = matQ @ (matK / math.sqrt(DH)).transpose(-2, -1)

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

    delta_f = delta_fbar * torch.sigmoid(-vecF)

    # delta_i: input gate preactivation delta errors
    delta_i = torch.sum(delta_Dtilde, dim=-2).unsqueeze_(-1)

    # output delta-errors / gradients

    delta_Q = (delta_C * var_D) @ (matK / math.sqrt(DH))
    delta_K = (delta_C * var_D).transpose(-2, -1) @ (matQ / math.sqrt(DH))

    var_C = var_QK * var_D
    delta_V = var_C.transpose(-2, -1) @ (matDeltaHtilde / (vecN + eps))
    return delta_Q, delta_K, delta_V, delta_i, delta_f, var_D
