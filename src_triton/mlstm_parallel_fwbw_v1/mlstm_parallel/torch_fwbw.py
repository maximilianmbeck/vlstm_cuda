# Copyright JKU Linz 2024
# Author: Maximilian Beck
import math

import torch
import torch.nn.functional as F

"""
PyTorch

mLSTM forward and backward pass. Parallel formulation.
"""


def _mlstm_fw(
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

    return matH, vecM.squeeze(-1), vecN.squeeze(-1)


def _mlstm_bw(
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
    )


def mlstm_fw(
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
    vecI: torch.Tensor,
    vecF: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    matH, _, _ = _mlstm_fw(matQ, matK, matV, vecI, vecF, eps)
    return matH


def mlstm_fwbw(
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
    vecI: torch.Tensor,
    vecF: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    matH, _, _ = _mlstm_fwbw.apply(matQ, matK, matV, vecI, vecF, eps)
    return matH


class _mlstm_fwbw(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        matQ: torch.Tensor,
        matK: torch.Tensor,
        matV: torch.Tensor,
        vecI: torch.Tensor,
        vecF: torch.Tensor,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        matH, vecM, vecN = _mlstm_fw(
            matQ=matQ,
            matK=matK,
            matV=matV,
            vecI=vecI,
            vecF=vecF,
            eps=eps,
        )
        ctx.save_for_backward(matQ, matK, matV, vecI, vecF, vecM, vecN)
        return matH, vecM, vecN

    @staticmethod
    def backward(
        ctx,
        matDeltaHtilde: torch.Tensor,
        vecDeltaM_unused: torch.Tensor,
        vecDeltaN_unused: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        (matQ, matK, matV, vecI, vecF, vecM, vecN) = ctx.saved_tensors
        matDeltaQ, matDeltaK, matDeltaV, vecDeltaI, vecDeltaF = _mlstm_bw(
            matDeltaHtilde=matDeltaHtilde,
            matQ=matQ,
            matK=matK,
            matV=matV,
            vecI=vecI,
            vecF=vecF,
            vecM=vecM,
            vecN=vecN,
        )
        return matDeltaQ, matDeltaK, matDeltaV, vecDeltaI, vecDeltaF, None
