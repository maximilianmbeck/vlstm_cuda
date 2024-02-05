# Copyright JKU Linz 2024
# Author: Maximilian Beck

import math

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


def vlstm_fw_nogatematrices_torch(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    stabilize_rowwise: bool = True,
    eps: float = 1e-6,
    **kwargs,
) -> torch.Tensor:
    """This is the core linear hopfield retrieval operation in parallel form.
    This version is stabilized. We control the range of exp() arguments by
    ensuring that they are always smaller than 0.0 by subtracting the maximum.

    Args:
        queries (torch.Tensor): (B, NH, S, DH)
        keys (torch.Tensor): (B, NH, S, DH)
        values (torch.Tensor): (B, NH, S, DH)
        igate_preact (torch.Tensor): (B, NH, S, S)
        fgate_preact (torch.Tensor): (B, NH, S, S)
        stabilize_rowwise (bool, optional): Wether to stabilize the combination matrix C rowwise (take maximum per row).
            Alternative: Subtract the maximum over all rows. Defaults to True.

    Returns:
        torch.Tensor: (B, NH, S, DH), retrieved values
    """

    B, NH, S, DH = queries.shape
    _dtype, _device = queries.dtype, queries.device

    # gate decay matrix D (combination of forget gate and input gate)
    log_D_matrix = fgate_preact + igate_preact  # (B, NH, S, S)
    # D matrix stabilization
    if stabilize_rowwise:
        max_log_D, _ = torch.max(log_D_matrix, dim=-1, keepdim=True)  # (B, NH, S, 1)
    else:
        max_log_D = torch.max(log_D_matrix.view(B, NH, -1), dim=-1, keepdim=True)[
            0
        ].unsqueeze(-1)
        # (B, NH, 1, 1)
    log_D_matrix_stabilized = log_D_matrix - max_log_D  # (B, NH, S, S)
    D_matrix = torch.exp(log_D_matrix_stabilized)  # (B, NH, S, S)

    keys = keys / math.sqrt(DH)

    # combination matrix C
    qk_matrix = queries @ keys.transpose(-2, -1)  # (B, NH, S, S)
    C_matrix = qk_matrix * D_matrix  # (B, NH, S, S)
    normalizer = torch.maximum(
        C_matrix.sum(dim=-1, keepdim=True).abs(), torch.exp(-max_log_D)
    )  # (B, NH, S, 1)
    # (B, NH, S, S)
    C_matrix_normalized = C_matrix / (normalizer + eps)

    # retrieved values
    retrieved_values = C_matrix_normalized @ values  # (B, NH, S, DH)
    return retrieved_values


def vlstm_fwbw_nogatematrices_torch(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    stabilize_rowwise: bool = True,
    eps: float = 1e-6,
    **kwargs,
) -> torch.Tensor:
    return vLSTMFwBwNoGateMatrices.apply(
        queries,
        keys,
        values,
        igate_preact,
        fgate_preact,
        stabilize_rowwise,
        eps,
    )


class vLSTMFwBwNoGateMatrices(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        igate_preact: torch.Tensor,
        fgate_preact: torch.Tensor,
        stabilize_rowwise: bool = True,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Args:
            queries (torch.Tensor): (B, NH, S, DH)
            keys (torch.Tensor): (B, NH, S, DH)
            values (torch.Tensor): (B, NH, S, DH)
            igate_preact (torch.Tensor): (B, NH, S, S)
            fgate_preact (torch.Tensor): (B, NH, S, S)
            stabilize_rowwise (bool, optional): Wether to stabilize the combination matrix C rowwise (take maximum per row).
                Alternative: Subtract the maximum over all rows. Defaults to True.
            eps (float, optional): Small constant to stabilize the division. Defaults to 1e-6.
        """
        B, NH, S, DH = queries.shape
        _dtype, _device = queries.dtype, queries.device

        var_Dtilde = fgate_preact + igate_preact
        if stabilize_rowwise:
            var_M, _ = torch.max(var_Dtilde, dim=-1, keepdim=True)
        else:
            var_M = torch.max(var_Dtilde.view(B, NH, -1), dim=-1, keepdim=True)[
                0
            ].unsqueeze(-1)

        var_D = torch.exp(var_Dtilde - var_M)

        keys = keys / math.sqrt(DH)

        var_QK = queries @ keys.transpose(-2, -1)

        var_Ctilde = var_QK * var_D

        var_B = var_Ctilde.sum(dim=-1, keepdim=True)

        var_N = torch.maximum(var_B.abs(), torch.exp(-var_M))

        var_C = var_Ctilde / (var_N + eps)

        var_R = var_C @ values

        ctx.save_for_backward(
            queries,
            keys,
            values,
            torch.tensor(eps, device=_device, dtype=_dtype),
            var_Ctilde,
            var_N,
            var_M,
            var_QK,
            var_D,
            var_C,
        )
        return var_R

    @staticmethod
    def backward(ctx, grad_var_R: torch.Tensor) -> tuple[torch.Tensor, ...]:
        queries, keys, values, eps, var_Ctilde, var_N, var_M, var_QK, var_D, var_C = (
            ctx.saved_tensors
        )
        B, NH, S, DH = queries.shape
        _dtype, _device = queries.dtype, queries.device

        # intermediate delta-errors
        delta_C = grad_var_R @ values.transpose(-2, -1)

        delta_N = (-delta_C * var_Ctilde * (1 / (torch.square(var_N) + eps))).sum(
            dim=-1, keepdim=True
        )

        var_sumCtilde = var_Ctilde.sum(dim=-1, keepdim=True)

        # delta_B_ = delta_N * var_sumCtilde / (torch.abs(var_sumCtilde) + eps)
        delta_B_ = delta_N * torch.sign(var_sumCtilde)
        delta_B = torch.where(
            torch.abs(var_sumCtilde) > torch.exp(-var_M),
            delta_B_,
            torch.tensor(0.0, dtype=_dtype, device=_device),
        )

        delta_Ctilde_C = delta_C / (var_N + eps)
        delta_Ctilde_B = (
            delta_B  # will be broadcasted automatically along last dimension
        )
        delta_Ctilde = delta_Ctilde_C + delta_Ctilde_B

        delta_D = delta_Ctilde * var_QK

        delta_Dtilde = delta_D * var_D

        # output delta-errors / gradients
        delta_F = delta_Dtilde
        delta_I = delta_Dtilde

        delta_Q = (delta_Ctilde * var_D) @ (keys)
        delta_K = (delta_Ctilde * var_D).transpose(-2, -1) @ (queries / math.sqrt(DH))
        delta_V = var_C.transpose(-2, -1) @ grad_var_R

        grad_var_q = delta_Q
        grad_var_k = delta_K
        grad_var_v = delta_V
        grad_var_igate = delta_I
        grad_var_fgate = delta_F
        return (
            grad_var_q,
            grad_var_k,
            grad_var_v,
            grad_var_igate,
            grad_var_fgate,
            None,
            None,
        )


def vlstm_fw_nogatematrices_nostabilization(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    temp_Ctilde: torch.Tensor,  # (S, S)
    temp_D: torch.Tensor,  # (S, S)
    temp_QK: torch.Tensor,  # (S, S)
    temp_N: torch.Tensor,  # (S, 1)
    temp_B: torch.Tensor,  # (S, 1)
    eps: float = 1e-6,
):
    B, NH, S, DH = queries.shape
    _dtype, _device = queries.dtype, queries.device

    var_Dtilde = fgate_preact + igate_preact

    var_D = torch.exp(var_Dtilde) + temp_D

    keys = keys / math.sqrt(DH)

    var_QK = queries @ keys.transpose(-2, -1) + temp_QK

    var_Ctilde = var_QK * var_D + temp_Ctilde

    var_B = var_Ctilde.sum(dim=-1, keepdim=True) + temp_B

    var_N = (
        torch.maximum(var_B.abs(), torch.tensor(1.0, dtype=_dtype, device=_device))
        + temp_N
    )

    var_C = var_Ctilde / (var_N + eps)

    var_R = var_C @ values
    return var_R


def vlstm_fwbw_nogatematrices_nostabilization(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    temp_Ctilde: torch.Tensor,  # (S, S)
    temp_D: torch.Tensor,  # (S, S)
    temp_QK: torch.Tensor,  # (S, S)
    temp_N: torch.Tensor,  # (S, 1)
    temp_B: torch.Tensor,  # (S, 1)
    eps: float = 1e-6,
):
    return vLSTMFwBwNoGateMatricesNoStabilization.apply(
        queries,
        keys,
        values,
        igate_preact,
        fgate_preact,
        temp_Ctilde,
        temp_D,
        temp_QK,
        temp_N,
        temp_B,
        eps,
    )


class vLSTMFwBwNoGateMatricesNoStabilization(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        igate_preact: torch.Tensor,
        fgate_preact: torch.Tensor,
        temp_Ctilde: torch.Tensor,  # (S, S)
        temp_D: torch.Tensor,  # (S, S)
        temp_QK: torch.Tensor,  # (S, S)
        temp_N: torch.Tensor,  # (S, 1)
        temp_B: torch.Tensor,  # (S, 1)
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Args:
            queries (torch.Tensor): (B, NH, S, DH)
            keys (torch.Tensor): (B, NH, S, DH)
            values (torch.Tensor): (B, NH, S, DH)
            igate_preact (torch.Tensor): (B, NH, S, S)
            fgate_preact (torch.Tensor): (B, NH, S, S)
            temp (torch.Tensor): (B, NH, S, S)
            stabilize_rowwise (bool, optional): Wether to stabilize the combination matrix C rowwise (take maximum per row).
                Alternative: Subtract the maximum over all rows. Defaults to True.
            eps (float, optional): Small constant to stabilize the division. Defaults to 1e-6.
        """
        B, NH, S, DH = queries.shape
        _dtype, _device = queries.dtype, queries.device

        var_Dtilde = fgate_preact + igate_preact

        var_D = torch.exp(var_Dtilde) + temp_D

        keys = keys / math.sqrt(DH)

        var_QK = queries @ keys.transpose(-2, -1) + temp_QK

        var_Ctilde = var_QK * var_D + temp_Ctilde

        var_B = var_Ctilde.sum(dim=-1, keepdim=True) + temp_B

        var_N = (
            torch.maximum(var_B.abs(), torch.tensor(1.0, dtype=_dtype, device=_device))
            + temp_N
        )

        var_C = var_Ctilde / (var_N + eps)

        var_R = var_C @ values

        ctx.save_for_backward(
            queries,
            keys,
            values,
            torch.tensor(eps, device=_device, dtype=_dtype),
            var_Ctilde,
            var_N,
            var_QK,
            var_D,
            var_C,
        )
        return var_R

    @staticmethod
    def backward(ctx, grad_var_R: torch.Tensor) -> tuple[torch.Tensor, ...]:
        queries, keys, values, eps, var_Ctilde, var_N, var_QK, var_D, var_C = (
            ctx.saved_tensors
        )
        B, NH, S, DH = queries.shape
        _dtype, _device = queries.dtype, queries.device

        # intermediate delta-errors
        delta_C = grad_var_R @ values.transpose(-2, -1)

        delta_N = (-delta_C * var_Ctilde * (1 / (torch.square(var_N) + eps))).sum(
            dim=-1, keepdim=True
        )

        var_sumCtilde = var_Ctilde.sum(dim=-1, keepdim=True)

        # delta_B_ = delta_N * var_sumCtilde / (torch.abs(var_sumCtilde) + eps)
        delta_B_ = delta_N * torch.sign(var_sumCtilde)
        delta_B = torch.where(
            torch.abs(var_sumCtilde) > 1.0,
            delta_B_,
            torch.tensor(0.0, dtype=_dtype, device=_device),
        )

        delta_Ctilde_C = delta_C / (var_N + eps)
        delta_Ctilde_B = (
            delta_B  # will be broadcasted automatically along last dimension
        )
        delta_Ctilde = delta_Ctilde_C + delta_Ctilde_B

        delta_D = delta_Ctilde * var_QK

        delta_Dtilde = delta_D * var_D

        # output delta-errors / gradients
        delta_F = delta_Dtilde
        delta_I = delta_Dtilde

        delta_Q = (delta_Ctilde * var_D) @ (keys)
        delta_K = (delta_Ctilde * var_D).transpose(-2, -1) @ (queries / math.sqrt(DH))
        delta_V = var_C.transpose(-2, -1) @ grad_var_R

        grad_var_q = delta_Q
        grad_var_k = delta_K
        grad_var_v = delta_V
        grad_var_igate = delta_I
        grad_var_fgate = delta_F
        grad_temp_Ctilde = delta_Ctilde
        grad_temp_D = delta_D
        grad_temp_QK = delta_Ctilde * var_D
        grad_temp_N = delta_N
        grad_temp_B = delta_B
        return (
            grad_var_q,
            grad_var_k,
            grad_var_v,
            grad_var_igate,
            grad_var_fgate,
            grad_temp_Ctilde,
            grad_temp_D,
            grad_temp_QK,
            grad_temp_N,
            grad_temp_B,
            None,
            None,
        )
