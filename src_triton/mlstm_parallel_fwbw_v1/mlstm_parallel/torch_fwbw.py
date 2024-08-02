# Copyright JKU Linz 2024
# Author: Maximilian Beck
import math

import torch
import torch.nn.functional as F

"""
PyTorch

mLSTM forward and backward pass. Parallel formulation.
"""


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


def mlstm_fwbw(
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
