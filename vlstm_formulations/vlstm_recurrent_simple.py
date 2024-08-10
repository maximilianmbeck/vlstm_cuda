# Copyright JKU Linz 2024
# Author: Maximilian Beck
import math

import torch
import torch.nn.functional as F

"""This module contains only the final recurrent implementation of the vLSTM."""


def vlstm_recurrent_sequence_stabilized(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    eps: float = 1e-6,
    **kwargs,
) -> torch.Tensor:
    """This is the core mLSTM operation in stabilized recurrent form. It operates on a full
    input sequence of length S. It is stabilized by adding a third "max" state.
    This is analog to [1].

    [1] Milakov, Maxim, and Natalia Gimelshein. “Online Normalizer Calculation for Softmax.” arXiv, July 28, 2018.
        http://arxiv.org/abs/1805.02867.


    Args:
        queries (torch.Tensor): (B, NH, S, DH)
        keys (torch.Tensor): (B, NH, S, DH)
        values (torch.Tensor): (B, NH, S, DH)
        igate_preact (torch.Tensor): (B, NH, S, 1)
        fgate_preact (torch.Tensor): (B, NH, S, 1)
        eps (float, optional): Used for building the forgetgate matrix. Defaults to 1e-6.

    Returns:
        torch.Tensor: (B, NH, S, DH), hidden states
    """

    B, NH, S, DH = queries.shape
    device = queries.device
    dtype = queries.dtype

    # memory state
    c_state = torch.zeros((B, NH, DH, DH), dtype=dtype, device=device)
    # normalizer state
    n_state = torch.zeros((B, NH, DH, 1), dtype=dtype, device=device)
    # max state
    m_state = torch.zeros((B, NH, 1, 1), dtype=dtype, device=device)

    hidden_states = []
    for t in range(S):
        # gates
        fg, ig = fgate_preact[:, :, t, :].unsqueeze(2), igate_preact[
            :, :, t, :
        ].unsqueeze(
            2
        )  # (B, NH, 1)
        # projections
        q, k, v = (
            queries[:, :, t, :].unsqueeze(2),
            keys[:, :, t, :].unsqueeze(2),
            values[:, :, t, :].unsqueeze(2),
        )  # (B, NH, DH)

        # step
        h, (c_state, n_state, m_state) = vlstm_recurrent_step_stabilized(
            c_state=c_state,
            n_state=n_state,
            m_state=m_state,
            q=q,
            k=k,
            v=v,
            igate_preact=ig,
            fgate_preact=fg,
            eps=eps,
        )
        hidden_states.append(h)

    hidden_states = torch.stack(hidden_states, dim=-2)  # (B, NH, S, DH)
    return hidden_states


def vlstm_recurrent_step_stabilized(
    c_state: torch.Tensor,
    n_state: torch.Tensor,
    m_state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    eps: float = 1e-6,
    **kwargs,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """This is a single step of the mLSTM operation in recurrent form.

    Args:
        c_state (torch.Tensor): (B, NH, DH, DH)
        n_state (torch.Tensor): (B, NH, DH, 1)
        m_state (torch.Tensor): (B, NH, 1, 1)
        q (torch.Tensor): (B, NH, 1, DH)
        k (torch.Tensor): (B, NH, 1, DH)
        v (torch.Tensor): (B, NH, 1, DH)
        igate_preact (torch.Tensor): (B, NH, 1, 1)
        fgate_preact (torch.Tensor): (B, NH, 1, 1)

    Returns:
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
            (hidden_state [B, NH, DH], (c_state_new [B, NH, DH, DH], n_state_new [B, NH, DH, 1]], m_state_new [B, NH, 1, 1]))
    """
    B, NH, S, DH = q.shape
    # projections
    q, k, v = (
        q.squeeze_(2).unsqueeze(-1),
        k.squeeze_(2).unsqueeze(-1),
        v.squeeze_(2).unsqueeze(-1),
    )  # (B, NH, DH, 1)

    # gates
    log_fg_act = torch.nn.functional.logsigmoid(fgate_preact)

    # update rule
    m_state_new = torch.max(log_fg_act + m_state, igate_preact)  # (B, NH, 1, 1)

    fg_act = torch.exp(log_fg_act + m_state - m_state_new)  # (B, NH, 1, 1)
    ig_act = torch.exp(igate_preact - m_state_new)  # (B, NH, 1, 1)

    k_scaled = k / math.sqrt(DH)

    c_state_new = fg_act * c_state + ig_act * (
        k_scaled @ v.transpose(-1, -2)
    )  # (B, NH, DH, DH)
    n_state_new = fg_act * n_state + ig_act * k_scaled  # (B, NH, DH, 1)

    h_num = q.transpose(-1, -2) @ c_state_new  # (B, NH, 1, DH)
    h_num = h_num.squeeze(-2)  # (B, NH, DH)

    qn_dotproduct = q.transpose(-1, -2) @ n_state_new  # (B, NH, 1, 1)
    max_val = torch.exp(-m_state_new)  # (B, NH, 1, 1)
    h_denom = torch.maximum(qn_dotproduct.abs(), max_val) + eps
    h_denom = h_denom.squeeze(2)  # (B, NH, 1)
    h = h_num / h_denom  # (B, NH, 1, DH) / (B, NH, 1, 1) = (B, NH, 1, DH)

    return h, (c_state_new, n_state_new, m_state_new)
