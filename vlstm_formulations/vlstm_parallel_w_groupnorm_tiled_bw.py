import math

import torch
import torch.nn.functional as F


def mlstm_parallel_w_groupnorm_torch_tiled_bw(
    matDeltaHtilde: torch.Tensor,
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
    vecI: torch.Tensor,
    vecF: torch.Tensor,
    vecM: torch.Tensor,
    vecN: torch.Tensor,
) -> tuple[torch.Tensor, ...]:

    pass


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
    return delta_Q, delta_K, delta_V, delta_i, delta_f
