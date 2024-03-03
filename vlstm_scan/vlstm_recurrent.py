import math
from typing import Callable

import torch

"""This file contains the vlstm kernels templates in pytorch for the parallel and recurrent form.
In this file all kernels have a exponential input gate and a sigmoid forget gate.

# B: batch_size
# S: sequence_length / context_length
# H: hidden_size
# NH: num_heads
# DH: head_dim = hidden_size / num_heads
"""


def log_sigmoid(x):
    return torch.where(
        x > 0.0, torch.log(torch.sigmoid(x)), x - torch.log(1.0 + torch.exp(x))
    )


def qs_normalizer_recurrent(
    qz_dotproduct: torch.Tensor,
    normalization_mode: str,
    eps: float = 1e-6,
    max_val: torch.Tensor = None,
) -> torch.Tensor:
    if normalization_mode in "sum_C":
        r_denom = qz_dotproduct + eps
    elif normalization_mode in "abs_sum_C":
        r_denom = qz_dotproduct.abs() + eps
    elif normalization_mode in "max_abs_sum_C_1":
        mval = (
            max_val
            if max_val is not None
            else torch.tensor(
                1.0, dtype=qz_dotproduct.dtype, device=qz_dotproduct.device
            )
        )
        r_denom = torch.maximum(qz_dotproduct.abs(), mval) + eps
    else:
        raise ValueError(
            f"normalization_mode must be one of ['sum_C', 'abs_sum_C'], got {normalization_mode}"
        )
    return r_denom


#### RECURRENT VERSIONS

# following normalizations are implemented and checked between parallel and recurrent form:
# sum_C: OK
# abs_sum_C: OK
# sum_abs_C: ERROR, #! cannot be implemented in recurrent form
# max_abs_sum_C_1:


def vlstm_recurrent_step(
    s_state: torch.Tensor,
    z_state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ig: torch.Tensor,
    fg: torch.Tensor,
    qk_decmask_normalize: bool = True,
    qk_dim_normalize: bool = True,
    normalization_mode: str = "max_abs_sum_C_1",
    eps: float = 1e-6,
    **kwargs,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """This is a single step of the core linear hopfield retrieval operation in recurrent form.
    It uses an exponential input gate. This version is not stabilized.

    Args:
        s_state (torch.Tensor): (B, NH, DH, DH)
        z_state (torch.Tensor): (B, NH, DH, 1)
        q (torch.Tensor): (B, NH, DH)
        k (torch.Tensor): (B, NH, DH)
        v (torch.Tensor): (B, NH, DH)
        ig (torch.Tensor): (B, NH, 1)
        fg (torch.Tensor): (B, NH, 1)
        qk_decmask_normalize (bool, optional): Wether to normalize the combination matrix C. Defaults to True.
        qk_dim_normalize (bool, optional): Wether to divide the qk matrix by sqrt(head_dim). Defaults to False.


    Returns:
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]: (retrieved [B, NH, DH], (s_next [B, NH, DH, DH], z_next [B, NH, DH, 1]]))
    """
    B, NH, DH = q.shape
    # projections
    q, k, v = q.unsqueeze(-1), k.unsqueeze(-1), v.unsqueeze(-1)  # (B, NH, DH, 1)

    # gates
    fg = fg.unsqueeze(-1)  # (B, NH, 1, 1)
    fg_act = torch.sigmoid(fg)

    ig = ig.unsqueeze(-1)  # (B, NH, 1, 1)
    ig_act = torch.exp(ig)

    if qk_dim_normalize:
        k = k / math.sqrt(DH)

    # update rule
    s_new = fg_act * s_state + ig_act * (k @ v.transpose(-1, -2))  # (B, NH, DH, DH)
    z_new = fg_act * z_state + ig_act * k  # (B, NH, DH, 1)
    if normalization_mode == "sum_abs_C":
        z_new = torch.abs(z_new)

    # retrieve
    r_num = q.transpose(-1, -2) @ s_new  # (B, NH, 1, DH)

    if qk_decmask_normalize:
        qz_dotproduct = q.transpose(-1, -2) @ z_new  # (B, NH, 1, 1)
        r_denom = qs_normalizer_recurrent(
            qz_dotproduct=qz_dotproduct, normalization_mode=normalization_mode, eps=eps
        )
        r = r_num / r_denom  # (B, NH, 1, DH) / (B, NH, 1, 1) = (B, NH, 1, DH)
    else:
        r = r_num

    return r.squeeze(-2), (s_new, z_new)


def vlstm_recurrent_sequence(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    qk_decmask_normalize: bool = True,
    qk_dim_normalize: bool = True,
    normalization_mode: str = "max_abs_sum_C_1",
    eps: float = 1e-6,
    step_fn: Callable = vlstm_recurrent_step,
    **kwargs,
) -> torch.Tensor:
    """This is the core linear hopfield retrieval operation in recurrent form. It operates on a full
    input sequence of length S.
    This version is not stabilized.

    Args:
        queries (torch.Tensor): (B, NH, S, DH)
        keys (torch.Tensor): (B, NH, S, DH)
        values (torch.Tensor): (B, NH, S, DH)
        igate_preact (torch.Tensor): (B, NH, S, 1)
        fgate_preact (torch.Tensor): (B, NH, S, 1)
        lower_triangular_matrix (torch.Tensor, optional): (S,S). Defaults to None.
        qk_decmask_normalize (bool, optional): Wether to normalize the combination matrix C. Defaults to True.
        qk_dim_normalize (bool, optional): Wether to divide the qk matrix by sqrt(head_dim). Defaults to False.
        eps (float, optional): Used for building the forgetgate matrix. Defaults to 1e-6.

    Returns:
        torch.Tensor: retrieved values, shape: (B, NH, S, DH)
    """

    B, NH, S, DH = queries.shape
    device = queries.device
    dtype = queries.dtype

    # memory state
    s = torch.zeros((B, NH, DH, DH), dtype=dtype, device=device)
    # normalizer state
    z = torch.zeros((B, NH, DH, 1), dtype=dtype, device=device)

    rs = []
    for t in range(S):
        # gates
        fg, ig = fgate_preact[:, :, t, :], igate_preact[:, :, t, :]  # (B, NH, 1)
        # projections
        q, k, v = (
            queries[:, :, t, :],
            keys[:, :, t, :],
            values[:, :, t, :],
        )  # (B, NH, DH)

        # step
        r, (s, z) = step_fn(
            s_state=s,
            z_state=z,
            q=q,
            k=k,
            v=v,
            ig=ig,
            fg=fg,
            qk_decmask_normalize=qk_decmask_normalize,
            qk_dim_normalize=qk_dim_normalize,
            normalization_mode=normalization_mode,
            eps=eps,
        )
        rs.append(r)

    rs = torch.stack(rs, dim=-2)  # (B, NH, S, DH)
    return rs


def vlstm_recurrent_sequence_stabilized(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    qk_decmask_normalize: bool = True,
    qk_dim_normalize: bool = True,
    normalization_mode: str = "max_abs_sum_C_1",
    eps: float = 1e-6,
    **kwargs,
) -> torch.Tensor:
    """This is the core linear hopfield retrieval operation in stabilized recurrent form. It operates on a full
    input sequence of length S. This version is stabilized by adding a third "max" state.
    This is analog to [1].

    [1] Milakov, Maxim, and Natalia Gimelshein. “Online Normalizer Calculation for Softmax.” arXiv, July 28, 2018.
        http://arxiv.org/abs/1805.02867.


    Args:
        queries (torch.Tensor): (B, NH, S, DH)
        keys (torch.Tensor): (B, NH, S, DH)
        values (torch.Tensor): (B, NH, S, DH)
        igate_preact (torch.Tensor): (B, NH, S, 1)
        fgate_preact (torch.Tensor): (B, NH, S, 1)
        lower_triangular_matrix (torch.Tensor, optional): (S,S). Defaults to None.
        qk_decmask_normalize (bool, optional): Wether to normalize the combination matrix C. Defaults to True.
        qk_dim_normalize (bool, optional): Wether to divide the qk matrix by sqrt(head_dim). Defaults to False.
        eps (float, optional): Used for building the forgetgate matrix. Defaults to 1e-6.

    Returns:
        torch.Tensor: (B, NH, S, DH), retrieved values
    """

    B, NH, S, DH = queries.shape
    device = queries.device
    dtype = queries.dtype

    # memory state
    s = torch.zeros((B, NH, DH, DH), dtype=dtype, device=device)
    # normalizer state
    z = torch.zeros((B, NH, DH, 1), dtype=dtype, device=device)
    # max state
    m = torch.zeros((B, NH, 1, 1), dtype=dtype, device=device)

    rs = []
    for t in range(S):
        # gates
        fg, ig = fgate_preact[:, :, t, :], igate_preact[:, :, t, :]  # (B, NH, 1)
        # projections
        q, k, v = (
            queries[:, :, t, :],
            keys[:, :, t, :],
            values[:, :, t, :],
        )  # (B, NH, DH)

        # step
        r, (s, z, m) = vlstm_recurrent_step_stabilized(
            s_state=s,
            z_state=z,
            m_state=m,
            q=q,
            k=k,
            v=v,
            ig=ig,
            fg=fg,
            qk_decmask_normalize=qk_decmask_normalize,
            qk_dim_normalize=qk_dim_normalize,
            normalization_mode=normalization_mode,
            eps=eps,
        )
        rs.append(r)

    rs = torch.stack(rs, dim=-2)  # (B, NH, S, DH)
    return rs


def vlstm_recurrent_step_stabilized(
    s_state: torch.Tensor,
    z_state: torch.Tensor,
    m_state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ig: torch.Tensor,
    fg: torch.Tensor,
    qk_decmask_normalize: bool = True,
    qk_dim_normalize: bool = True,
    normalization_mode: str = "max_abs_sum_C_1",
    eps: float = 1e-6,
    **kwargs,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """This is a single step of the core linear hopfield retrieval operation in recurrent form.
    This version is not stabilized.

    Args:
        s_state (torch.Tensor): (B, NH, DH, DH)
        z_state (torch.Tensor): (B, NH, DH, 1)
        m_state (torch.Tensor): (B, NH, 1, 1)
        q (torch.Tensor): (B, NH, DH)
        k (torch.Tensor): (B, NH, DH)
        v (torch.Tensor): (B, NH, DH)
        ig (torch.Tensor): (B, NH, 1)
        fg (torch.Tensor): (B, NH, 1)
        qk_decmask_normalize (bool, optional): Wether to normalize the combination matrix C. Defaults to True.
        qk_dim_normalize (bool, optional): Wether to divide the qk matrix by sqrt(head_dim). Defaults to False.

    Returns:
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
            (retrieved [B, NH, DH], (s_next [B, NH, DH, DH], z_next [B, NH, DH, 1]], m_next [B, NH, 1, 1]))
    """
    B, NH, DH = q.shape
    # projections
    q, k, v = q.unsqueeze(-1), k.unsqueeze(-1), v.unsqueeze(-1)  # (B, NH, DH, 1)

    # gates
    fg = fg.unsqueeze(-1)  # (B, NH, 1, 1)
    log_fg_act = log_sigmoid(fg)

    ig = ig.unsqueeze(-1)  # (B, NH, 1, 1)

    # update rule
    m_new = torch.max(log_fg_act + m_state, ig)  # (B, NH, 1, 1)

    fg_act = torch.exp(log_fg_act + m_state - m_new)  # (B, NH, 1, 1)
    ig_act = torch.exp(ig - m_new)  # (B, NH, 1, 1)

    if qk_dim_normalize:
        k = k / math.sqrt(DH)

    s_new = fg_act * s_state + ig_act * (k @ v.transpose(-1, -2))  # (B, NH, DH, DH)
    z_new = fg_act * z_state + ig_act * k  # (B, NH, DH, 1)
    # if normalization_mode == "sum_abs_C":  #! ERROR: sum_abs_C not possible in recurrent form
    #     z_new = torch.abs(z_new)

    # retrieve
    r_num = q.transpose(-1, -2) @ s_new  # (B, NH, 1, DH)

    if qk_decmask_normalize:
        qz_dotproduct = q.transpose(-1, -2) @ z_new  # (B, NH, 1, 1)
        max_val = torch.exp(-m_new)  # (B, NH, 1, 1)
        r_denom = qs_normalizer_recurrent(
            qz_dotproduct=qz_dotproduct,
            normalization_mode=normalization_mode,
            eps=eps,
            max_val=max_val,
        )
        r = r_num / r_denom  # (B, NH, 1, DH) / (B, NH, 1, 1) = (B, NH, 1, DH)
    else:
        r = r_num

    return r.squeeze(-2), (s_new, z_new, m_new)
