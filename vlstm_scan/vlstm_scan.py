import math

import torch

"""In this module we implement the scan version of the VLSTM model."""


def log_sigmoid(x):
    return torch.where(
        x > 0.0, torch.log(torch.sigmoid(x)), x - torch.log(1.0 + torch.exp(x))
    )


## Reimplement the recurrent version as reference for the scan version.


def vlstm_scan_step_stabilized(
    c_state: torch.Tensor,
    n_state: torch.Tensor,
    m_state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ig: torch.Tensor,
    fg: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """vLSTM step function with stabilized gating mechanism.

    Args:
        c_state (torch.Tensor): (B, NH, DH, DH)
        n_state (torch.Tensor): (B, NH, DH, 1)
        m_state (torch.Tensor): (B, NH, 1, 1)
        q (torch.Tensor): (B, NH, DH)
        k (torch.Tensor): (B, NH, DH)
        v (torch.Tensor): (B, NH, DH)
        ig (torch.Tensor): (B, NH, 1)
        fg (torch.Tensor): (B, NH, 1)

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (c_state_next, n_state_next, m_state_next)
    """
    B, NH, DH = q.shape

    # projections
    q, k, v = q.unsqueeze(-1), k.unsqueeze(-1), v.unsqueeze(-1)

    # gates
    ig = ig.unsqueeze(-1)  # (B, NH, 1, 1)
    fg = fg.unsqueeze(-1)  # (B, NH, 1, 1)
    log_fg_act = log_sigmoid(fg)

    # update rule
    m_new = torch.max(log_fg_act + m_state, ig)  # (B, NH, 1, 1)

    fg_act = torch.exp(log_fg_act + m_state - m_new)  # (B, NH, 1, 1)
    ig_act = torch.exp(ig - m_new)  # (B, NH, 1, 1)
    k = k / math.sqrt(DH)

    c_new = fg_act * c_state + ig_act * (k @ v.transpose(-1, -2))  # (B, NH, DH, DH)
    n_new = fg_act * n_state + ig_act * k  # (B, NH, DH, 1)
