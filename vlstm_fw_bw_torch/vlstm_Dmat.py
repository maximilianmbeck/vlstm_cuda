# Copyright JKU Linz 2024
# Author: Maximilian Beck

import torch


def log_sigmoid(x):
    return torch.where(
        x > 0.0, torch.log(torch.sigmoid(x)), x - torch.log(1.0 + torch.exp(x))
    )


def vlstm_fw_Dtildemat(
    igate_preact: torch.Tensor, fgate_preact: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
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

    dmat = log_fg_matrix + ig_matrix
    return dmat


def vlstm_fwbw_Dtildemat(
    igate_preact: torch.Tensor, fgate_preact: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return vLSTMFwBwDtildemat.apply(igate_preact, fgate_preact)


class vLSTMFwBwDtildemat(torch.autograd.Function):

    @staticmethod
    def forward(ctx, igate_preact, fgate_preact):
        ctx.save_for_backward(fgate_preact)
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

        dmat = log_fg_matrix + ig_matrix
        return dmat

    @staticmethod
    def backward(ctx, grad_dtilde_mat):
        (fgate_preact,) = ctx.saved_tensors
        B, NH, S, _ = grad_dtilde_mat.shape
        _dtype, _device = grad_dtilde_mat.dtype, grad_dtilde_mat.device

        # delta_f: forget gate preactivation delta errors
        ltr_dm1 = torch.tril(
            torch.ones(
                (S, S),
                dtype=torch.bool,
                device=_device,
            ),
            diagonal=-1,  #! Also mask out the diagonal as it is constant 1 in the D matrix
        )
        masked_grad_dtilde_mat = torch.where(
            ltr_dm1,
            grad_dtilde_mat,
            torch.tensor(0.0, device=_device, dtype=_dtype),
        )

        delta_fbar = torch.zeros((B, NH, S, 1), device=_device, dtype=_dtype)
        # # first forget gate index (k=0) does not get a gradient (since it is not used in the forward pass)
        # TODO implement this in a more efficient way
        for k in range(1, S):
            for j in range(k):
                delta_fbar[:, :, k, 0] += (
                    masked_grad_dtilde_mat[:, :, k:, j].view(B, NH, -1).sum(dim=-1)
                )

        delta_f = delta_fbar * torch.sigmoid(-fgate_preact)

        # delta_i: input gate preactivation delta errors
        delta_i = torch.sum(grad_dtilde_mat, dim=-2).unsqueeze_(-1)

        grad_igate_preact = delta_i
        grad_fgate_preact = delta_f

        return grad_igate_preact, grad_fgate_preact
