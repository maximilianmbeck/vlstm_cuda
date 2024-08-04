# Copyright JKU Linz 2024
# Author: Maximilian Beck

import math

import torch
import torch.nn.functional as F
from einops import rearrange

"""This file contains the first mlstm chunkwise parallel implementation.

This is the starting point for the next iterations.
"""


def mlstm_chunkwise_parallel_legacy(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    chunk_size: int = 64,
):
    B, NH, S, DH = queries.shape
    _dtype, _device = queries.dtype, queries.device
    qs = rearrange(queries, "b nh (nc l) dh -> b nh nc l dh", l=chunk_size) * (DH**-0.5)
    ks = rearrange(keys, "b nh (nc l) dh -> b nh nc l dh", l=chunk_size)
    vs = rearrange(values, "b nh (nc l) dh -> b nh nc l dh", l=chunk_size)
    _, _, NC, L, _ = qs.shape
    igs = rearrange(igate_preact, "b nh (nc l) 1 -> b nh nc l", l=chunk_size)
    fgs = rearrange(fgate_preact, "b nh (nc l) 1 -> b nh nc l", l=chunk_size)

    # compute the gates, the g and the p and q vectors
    log_fgates = F.logsigmoid(fgs)  # fgs
    # print(f"log_fgates: {log_fgates.shape}\n{log_fgates}")

    p_vec_f = log_fgates[:, :, :, :].cumsum(-1)
    # print(f"p_vec_f: {p_vec_f.shape}\n{p_vec_f}")

    q_vec_f = log_fgates[:, :, :, :].sum(-1, keepdim=True) - p_vec_f  # q_vec_f_raw
    # print(f"q_vec_f: {q_vec_f.shape}\n{q_vec_f}")

    p_vec = p_vec_f
    q_vec = q_vec_f + igs
    g_vec = log_fgates.sum(-1)
    # print(f"g_vec: {g_vec.shape}\n{g_vec}")

    # get the maximum values per chunk for p and q
    p_vec_max = p_vec.max(-1).values
    q_vec_max = q_vec.max(-1).values

    # loop 1: materialize the  C_k, n_k, m_k
    C_states = torch.zeros((B, NH, NC, DH, DH), dtype=_dtype, device=_device)
    n_states = torch.zeros((B, NH, NC, DH), dtype=_dtype, device=_device)
    m_states = torch.zeros((B, NH, NC, 1), dtype=_dtype, device=_device)

    m_k = torch.zeros((B, NH, 1), dtype=_dtype, device=_device)
    m_prev_k = torch.zeros((B, NH, 1), dtype=_dtype, device=_device)
    C_k = torch.zeros((B, NH, DH, DH), dtype=_dtype, device=_device)
    C_prev_k = torch.zeros((B, NH, DH, DH), dtype=_dtype, device=_device)
    n_k = torch.zeros((B, NH, DH), dtype=_dtype, device=_device)
    n_prev_k = torch.zeros((B, NH, DH), dtype=_dtype, device=_device)
    for k in range(1, NC):
        i = k - 1
        # m_k
        m_q_k = q_vec_max[:, :, i]
        g_k = g_vec[:, :, i]
        m_k = torch.max(g_k + m_prev_k, m_q_k)
        m_states[:, :, k] = m_k

        # C_k
        k_chunk = ks[:, :, i, :, :].clone()
        v_chunk = vs[:, :, i, :, :].clone()
        q_k = q_vec[:, :, i, :].clone()
        k_chunk_gated = k_chunk * torch.exp(q_k - m_k).unsqueeze(-1)

        C_k = (
            torch.exp(g_k + m_prev_k - m_k) * C_prev_k
            + k_chunk_gated.transpose(-2, -1) @ v_chunk
        )
        C_states[:, :, k] = C_k

        # n_k
        n_k = torch.exp(g_k + m_prev_k - m_k) * n_prev_k + k_chunk_gated.transpose(
            -2, -1
        ).sum(-1)
        n_states[:, :, k] = n_k

        # move to the next iteration
        m_prev_k = m_k
        C_prev_k = C_k
        n_prev_k = n_k

    # loop 2: compute the H_states
    H_states = torch.zeros((B, NH, NC, L, DH), dtype=_dtype, device=_device)
    for k in range(1, NC + 1):
        i = k - 1

        # load C_k, n_k, m_k
        C_k = C_states[:, :, i]
        n_k_inter = n_states[:, :, i]
        m_k = m_states[:, :, i]
        # load q, k, v chunks
        q_chunk = qs[:, :, i, :, :].clone()
        k_chunk = ks[:, :, i, :, :].clone()
        v_chunk = vs[:, :, i, :, :].clone()

        # ? Compute intra chunk contribution: H_intra
        # this is similar to the parallel version, but only for the current chunk
        log_fg_k = log_fgates[:, :, i].unsqueeze(-1)  # (B, NH, L, 1)
        log_ig_k = igs[:, :, i].unsqueeze(-1)  # (B, NH, L, 1)
        ltr = torch.tril(
            torch.ones(
                (L, L),
                dtype=torch.bool,
                device=_device,
            )
        )
        log_fg_k_cumsum = torch.cat(
            [
                torch.zeros((B, NH, 1, 1), dtype=_dtype, device=_device),
                torch.cumsum(log_fg_k, dim=-2),
            ],
            dim=-2,
        )  # (B, NH, L+1, 1)

        # for each batch/head this is a matrix of shape (L+1, L+1) containing the cumsum of the log forget gate values
        # in the second dimension (colum dimension). Each row has the same is a copy of the first row.
        # First entry of each row is zero.
        rep_log_fg_k_cumsum = log_fg_k_cumsum.repeat(
            1, 1, 1, L + 1
        )  # (B, NH, L+1, L+1)
        # Now in each row cut off / subtract the forgetgate values of the later timesteps
        # where col j > row i
        _log_fg_k_matrix = rep_log_fg_k_cumsum - rep_log_fg_k_cumsum.transpose(
            -2, -1
        )  # (B, NH, L+1, L+1)
        # Causal masking & selection of the correct submatrix, such that forgetgate at timestep t is not applied
        # to the input at timestep t
        log_fg_k_matrix = torch.where(
            ltr, _log_fg_k_matrix[:, :, 1:, 1:], -float("inf")
        )  # (B, NH, L, L)
        # print(f"log_fg_k_matrix: {log_fg_k_matrix.shape}\n{log_fg_k_matrix}")
        log_D_k = log_fg_k_matrix + log_ig_k.transpose(-2, -1)  # (B, NH, L, L)

        # H_intra
        # max_state intra
        m_log_D_k = torch.max(log_D_k, dim=-1, keepdim=True).values

        # max_state inter
        p_k = p_vec[:, :, i, :].unsqueeze(-1)
        # m_k

        # max_state combined
        m_state_combined = torch.maximum(p_k + m_k, m_log_D_k)

        log_D_k_stabilized = log_D_k - m_state_combined
        D_k = torch.exp(log_D_k_stabilized)
        qk_k_matrix = q_chunk @ k_chunk.transpose(-2, -1)
        C_k_matrix = qk_k_matrix * D_k

        q_chunk_gated = q_chunk * torch.exp(p_k + m_k - m_state_combined)
        numerator_common = q_chunk_gated @ C_k + C_k_matrix @ v_chunk

        denom_common = q_chunk_gated @ n_k_inter.unsqueeze(-1) + C_k_matrix.sum(
            dim=-1, keepdim=True
        )

        H_k_state = numerator_common / torch.maximum(
            torch.abs(denom_common), torch.exp(-m_state_combined)
        )
        H_states[:, :, i, :, :] = H_k_state

    H_out = rearrange(H_states, "b nh nc l dh -> b nh (nc l) dh")
    return H_out
