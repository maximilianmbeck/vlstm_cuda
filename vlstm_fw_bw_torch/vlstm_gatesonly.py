# Copyright JKU Linz 2024
# Author: Maximilian Beck

import torch


def log_sigmoid(x):
    return torch.where(
        x > 0.0, torch.log(torch.sigmoid(x)), x - torch.log(1.0 + torch.exp(x))
    )


def vlstm_fw_prepare_gate_preacts(
    igate_preact: torch.Tensor, fgate_preact: torch.Tensor
) -> tuple[torch.Tensor]:
    B, NH, S, _ = fgate_preact.shape
    _dtype, _device = fgate_preact.dtype, fgate_preact.device

    # forget gate matrix
    log_fgates = log_sigmoid(fgate_preact)  # (B, NH, S, 1)
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

    return log_fg_matrix, ig_matrix


def vlstm_fwbw_prepare_gate_preacts(
    igate_preact: torch.Tensor, fgate_preact: torch.Tensor
):
    pass


class vLSTMFwBwPrepareGatePreacts(torch.autograd.Function):

    @staticmethod
    def forward(ctx, igate_preact, fgate_preact):

        B, NH, S, _ = fgate_preact.shape
        _dtype, _device = fgate_preact.dtype, fgate_preact.device

        # forget gate matrix
        log_fgates = log_sigmoid(fgate_preact)  # (B, NH, S, 1)
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

        ctx.save_for_backward(ltr)
        return log_fg_matrix, ig_matrix

    @staticmethod
    def backward(ctx, grad_fg, grad_ig):
        ltr = ctx.saved_tensors

        return None, None
