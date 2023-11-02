import torch


def qkv_vlstm_kernel_pytorch(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    D_matrix: torch.Tensor,
    mval: torch.Tensor,
    eps: float = 1e-6,
    qk_decmask_normalize: bool = True,
) -> torch.Tensor:
    """
    Args:
        queries (torch.Tensor): (B, NH, S, DH)
        keys (torch.Tensor): (B, NH, S, DH)
        values (torch.Tensor): (B, NH, S, DH)
        D_matrix (torch.Tensor): (B, NH, S, S)
        mval (torch.Tensor): (B, NH, 1, 1)
        eps (float, optional): Defaults to 1e-6.
        qk_decmask_normalize (bool, optional): Defaults to True.
    """

    # combination matrix C
    qk_matrix = queries @ keys.transpose(-2, -1)  # (B, NH, S, S)
    C_matrix = qk_matrix * D_matrix  # (B, NH, S, S)
    if qk_decmask_normalize:
        # (B, NH, S, S)
        normalizer = torch.maximum(C_matrix.sum(dim=-1, keepdim=True).abs(), mval)
        C_matrix_normalized = C_matrix / (normalizer + eps)
    else:
        C_matrix_normalized = C_matrix

    # retrieved values
    retrieved_values = C_matrix_normalized @ values  # (B, NH, S, DH)
    return retrieved_values


def qkv_vlstm_no_dmatrix_kernel_pytorch(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mval: torch.Tensor = None,
    eps: float = 1e-6,
    qk_decmask_normalize: bool = True,
) -> torch.Tensor:
    """
    Args:
        queries (torch.Tensor): (B, NH, S, DH)
        keys (torch.Tensor): (B, NH, S, DH)
        values (torch.Tensor): (B, NH, S, DH)
        mval (torch.Tensor): (B, NH, 1, 1)
        eps (float, optional): Defaults to 1e-6.
        qk_decmask_normalize (bool, optional): Defaults to True.
    """

    # combination matrix C
    qk_matrix = queries @ keys.transpose(-2, -1)  # (B, NH, S, S)
    C_matrix = qk_matrix  # (B, NH, S, S)
    if qk_decmask_normalize:
        assert mval is not None, "mval should be provided"
        # (B, NH, S, S)
        normalizer = torch.maximum(C_matrix.sum(dim=-1, keepdim=True).abs(), mval)
        C_matrix_normalized = C_matrix / (normalizer + eps)
    else:
        C_matrix_normalized = C_matrix

    # retrieved values
    retrieved_values = C_matrix_normalized @ values  # (B, NH, S, DH)
    return retrieved_values
