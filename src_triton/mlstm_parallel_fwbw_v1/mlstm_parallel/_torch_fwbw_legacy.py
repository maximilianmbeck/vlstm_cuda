# Copyright JKU Linz 2024
# Author: Maximilian Beck
import math

import torch
import torch.nn.functional as F

"""
PyTorch

mLSTM forward and backward pass. Parallel formulation.
"""


def mlstm_fw_legacy(
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
    vecI: torch.Tensor,
    vecF: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """This is the legacy version that was used in the paper."""

    B, NH, S, DH = matQ.shape
    _dtype, _device = matQ.dtype, matQ.device

    # forget gate matrix
    log_fgates = F.logsigmoid(vecF)  # (B, NH, S, 1)
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
    log_D_matrix = log_fg_matrix + vecI.transpose(-2, -1)  # (B, NH, S, S)

    # Debugging only: f_gate only:
    # log_D_matrix = log_fg_matrix  # (B, NH, S, S)
    # D matrix stabilization
    max_log_D, _ = torch.max(log_D_matrix, dim=-1, keepdim=True)  # (B, NH, S, 1)
    log_D_matrix_stabilized = log_D_matrix - max_log_D  # (B, NH, S, S)
    D_matrix = torch.exp(log_D_matrix_stabilized)  # (B, NH, S, S)

    keys_scaled = matK / math.sqrt(DH)

    # combination matrix C
    qk_matrix = matQ @ keys_scaled.transpose(-2, -1)  # (B, NH, S, S)
    C_matrix = qk_matrix * D_matrix  # (B, NH, S, S)
    l = C_matrix.sum(dim=-1, keepdim=True)
    normalizer = torch.maximum(l.abs(), torch.exp(-max_log_D))  # (B, NH, S, 1)
    # (B, NH, S, S)
    C_matrix_normalized = C_matrix / (normalizer + eps)

    # retrieved values
    retrieved_values = C_matrix_normalized @ matV  # (B, NH, S, DH)
    return retrieved_values, max_log_D, normalizer, log_fg_matrix


def mlstm_fw(
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
    vecI: torch.Tensor,
    vecF: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:

    B, NH, S, DH = matQ.shape
    assert matK.shape == (B, NH, S, DH)
    assert matV.shape == (B, NH, S, DH)
    assert vecI.shape == (B, NH, S)
    assert vecF.shape == (B, NH, S)

    _dtype, _device = matQ.dtype, matQ.device

    vecLogSigF = F.logsigmoid(vecF)  # (B, NH, S)
    vecLogSigF_cumsum = vecLogSigF.cumsum(-1)

    matLogSigF = vecLogSigF_cumsum[:, :, :, None] - vecLogSigF_cumsum[:, :, None, :]

    ltr = torch.tril(
        torch.ones(
            (S, S),
            dtype=torch.bool,
            device=_device,
        )
    )

    matLogSigF_mask = torch.where(ltr, matLogSigF, -float("inf"))

    matLogD = matLogSigF_mask + vecI[:, :, None, :]

    vecM, _ = torch.max(matLogD, dim=-1, keepdim=True)  # (B, NH, S, 1)
    matLogD_stabilized = matLogD - vecM

    matD = torch.exp(matLogD_stabilized)  # (B, NH, S, S)

    matS = (matQ @ matK.transpose(-2, -1)) / math.sqrt(DH)  # (B, NH, S, S)

    matCtilde = matS * matD  # (B, NH, S, S)
    vecN = torch.maximum(
        matCtilde.sum(dim=-1, keepdim=True).abs(), torch.exp(-vecM)
    )  # (B, NH, S, 1)
    # (B, NH, S, S)
    matC = matCtilde / (vecN + eps)

    matH = matC @ matV  # (B, NH, S, DH)

    return matH, vecM.squeeze(-1), vecN.squeeze(-1), matLogSigF_mask


def mlstm_bw_legacy(
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
    var_D = torch.exp(var_Dtilde - vecM).to(dtype=_dtype)

    # intermediate delta-errors
    delta_C = matDeltaHtilde @ matV.transpose(-2, -1) / (vecN + eps)

    var_QK = matQ @ (matK / math.sqrt(DH)).transpose(-2, -1)

    delta_D = delta_C * var_QK

    delta_Dtilde = delta_D * var_D

    # FGATE VARIANT 1: NAIVE
    # # compute fgate and igate preact delta errors
    # # delta_f: forget gate preactivation delta errors
    # ltr_dm1 = torch.tril(
    #     torch.ones(
    #         (S, S),
    #         dtype=torch.bool,
    #         device=_device,
    #     ),
    #     diagonal=-1,  #! Also mask out the diagonal as it is constant 1 in the D matrix
    # )
    # masked_deltaDtilde = torch.where(
    #     ltr_dm1,
    #     delta_Dtilde,
    #     torch.tensor(0.0, device=_device, dtype=_dtype),
    # )

    # delta_fbar = torch.zeros((B, NH, S, 1), device=_device, dtype=_dtype)
    # # # first forget gate index (k=0) does not get a gradient (since it is not used in the forward pass)
    # for k in range(1, S):
    #     for j in range(k):
    #         delta_fbar[:, :, k, 0] += (
    #             masked_deltaDtilde[:, :, k:, j].view(B, NH, -1).sum(dim=-1)
    #         )
    # # more efficient way would be
    # # delta_fbar = delta_Dtilde.cumsum(-1).tril(-1).sum(dim=-2)

    # delta_f = delta_fbar * torch.sigmoid(-vecF)
    # #! DEBUG only
    # # delta_f = delta_fbar

    # delta_i: input gate preactivation delta errors
    delta_i = torch.sum(delta_Dtilde, dim=-2).unsqueeze_(-1)

    # output delta-errors / gradients

    delta_Q = (delta_C * var_D) @ (matK / math.sqrt(DH))
    delta_K = (delta_C * var_D).transpose(-2, -1) @ (matQ / math.sqrt(DH))

    var_C = var_QK * var_D
    delta_V = var_C.transpose(-2, -1) @ (matDeltaHtilde / (vecN + eps))

    # FGATE VARIANT 3: EFFICIENT LINEAR ATTENTION TRICK
    # compute the vecDeltaFbar values with dfbar = rev_cumsum((q*dq - k*dk).sum(-1))
    vecDeltaFbar_acc = (matQ * delta_Q - matK * delta_K).sum(-1)
    vecDeltaFbar = vecDeltaFbar_acc.flip(-1).cumsum(-1).flip(-1)
    vecDeltaF = vecDeltaFbar * torch.sigmoid(-vecF.squeeze(-1))
    delta_f = vecDeltaF.unsqueeze(-1)

    return (
        delta_Q,
        delta_K,
        delta_V,
        delta_i,
        delta_f,
        delta_C,
        delta_Dtilde,
        var_D,
        var_C,
    )


def mlstm_bw(
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
    assert matK.shape == (B, NH, S, DH)
    assert matV.shape == (B, NH, S, DH)
    assert vecI.shape == (B, NH, S)
    assert vecF.shape == (B, NH, S)

    _dtype, _device = matQ.dtype, matQ.device

    vecLogSigF = F.logsigmoid(vecF)  # (B, NH, S)
    vecLogSigF_cumsum = vecLogSigF.cumsum(-1)

    matLogSigF = vecLogSigF_cumsum[:, :, :, None] - vecLogSigF_cumsum[:, :, None, :]

    ltr = torch.tril(
        torch.ones(
            (S, S),
            dtype=torch.bool,
            device=_device,
        )
    )

    matLogSigF_mask = torch.where(ltr, matLogSigF, -float("inf"))

    matLogD = matLogSigF_mask + vecI[:, :, None, :]

    matLogD_stabilized = matLogD - vecM[:, :, :, None]

    matD = torch.exp(matLogD_stabilized)  # (B, NH, S, S)

    # intermediate delta-errors
    matDeltaC = matDeltaHtilde @ matV.transpose(-2, -1) / (vecN[:, :, :, None] + eps)

    matS = (matQ @ matK.transpose(-2, -1)) / math.sqrt(DH)

    # matDeltaD = matDeltaC * matS

    matDeltaDtilde = matDeltaC * matD * matS

    vecDeltaI = torch.sum(matDeltaDtilde, dim=-2)

    # output delta-errors / gradients
    matP = matDeltaC * matD

    matDeltaQ = (matP @ matK) / math.sqrt(DH)
    matDeltaK = (matP.transpose(-2, -1) @ matQ) / math.sqrt(DH)

    matCtilde = matS * matD
    matDeltaV = matCtilde.transpose(-2, -1) @ (
        matDeltaHtilde / (vecN[:, :, :, None] + eps)
    )

    # EFFICIENT LINEAR ATTENTION TRICK
    # compute the vecDeltaFbar values with dfbar = rev_cumsum((q*dq - k*dk).sum(-1))
    vecDeltaFbar_acc = (matQ * matDeltaQ - matK * matDeltaK).sum(-1)
    vecDeltaFbar = vecDeltaFbar_acc.flip(-1).cumsum(-1).flip(-1)
    vecDeltaF = vecDeltaFbar * torch.sigmoid(-vecF)

    return (
        matDeltaQ,
        matDeltaK,
        matDeltaV,
        vecDeltaI,
        vecDeltaF,
        matDeltaC,
        matDeltaDtilde,
        matD,
        matCtilde,
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
        delta_Q, delta_K, delta_V, delta_i, delta_f, _, _, _, _, _ = (
            vlstm_parallel_w_groupnorm_torch_bw(
                matDeltaHtilde=delta_Htilde,
                matQ=queries,
                matK=keys,
                matV=values,
                vecI=igate_preact,
                vecF=fgate_preact,
                vecN=var_n,
                vecM=var_m,
            )
        )
        return delta_Q, delta_K, delta_V, delta_i, delta_f, None
