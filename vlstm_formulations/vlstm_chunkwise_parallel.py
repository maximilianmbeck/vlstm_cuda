# Copyright JKU Linz 2024
# Author: Maximilian Beck

import math

import torch
import torch.nn.functional as F
from einops import rearrange

# #! Not working, errors in the implementation
# def vlstm_chunkwise_parallel_1(
#     queries: torch.Tensor,
#     keys: torch.Tensor,
#     values: torch.Tensor,
#     igate_preact: torch.Tensor,
#     fgate_preact: torch.Tensor,
#     chunk_size: int = 64,
# ):
#     B, NH, S, DH = queries.shape
#     _dtype, _device = queries.dtype, queries.device
#     qs = rearrange(queries, "b nh (nc l) dh -> b nh nc l dh", l=chunk_size) * (DH**-0.5)
#     ks = rearrange(keys, "b nh (nc l) dh -> b nh nc l dh", l=chunk_size)
#     vs = rearrange(values, "b nh (nc l) dh -> b nh nc l dh", l=chunk_size)
#     _, _, NC, L, _ = qs.shape
#     igs = rearrange(igate_preact, "b nh (nc l) 1 -> b nh nc l", l=chunk_size)
#     fgs = rearrange(fgate_preact, "b nh (nc l) 1 -> b nh nc l", l=chunk_size)
#     log_fgates = F.logsigmoid(fgs)
#     # print(ks.shape)
#     # print(log_fgates.shape)

#     # compute the g_vec:
#     g_vec = log_fgates.sum(-1)

#     # compute the d_vec:
#     d_vec_raw = torch.cat(
#         [
#             torch.zeros((B, NH, NC, 1), dtype=_dtype, device=_device),
#             log_fgates[:, :, :, 1:].cumsum(-1),
#         ],
#         dim=-1,
#     )
#     # unsqueeze the last dimension (head dim) in order to allow broadcasting
#     d_vec = (d_vec_raw + igs).unsqueeze(-1)
#     # print(d_vec.shape)

#     m_state = torch.zeros((B, NH, 1), dtype=_dtype, device=_device)
#     m_prev_state = torch.zeros((B, NH, 1), dtype=_dtype, device=_device)
#     C_state = torch.zeros((B, NH, DH, DH), dtype=_dtype, device=_device)
#     C_prev_state = torch.zeros((B, NH, DH, DH), dtype=_dtype, device=_device)
#     n_state = torch.zeros((B, NH, DH), dtype=_dtype, device=_device)
#     n_prev_state = torch.zeros((B, NH, DH), dtype=_dtype, device=_device)
#     # get the maximum values per chunk
#     m_k1_perchunk = d_vec.max(-2).values
#     # apply the d_vec to the keys
#     ks_gated = ks * d_vec

#     kv_gated = ks_gated.transpose(-2, -1) @ vs

#     C_states = torch.zeros((B, NH, NC, DH, DH), dtype=_dtype, device=_device)
#     n_states = torch.zeros((B, NH, NC, DH), dtype=_dtype, device=_device)
#     H_states = torch.zeros((B, NH, NC, L, DH), dtype=_dtype, device=_device)

#     for k in range(0, NC):
#         # print(k)
#         m_k1 = m_k1_perchunk[:, :, k]
#         g_state = g_vec[:, :, k]
#         m_state = torch.max(g_state + m_prev_state, m_k1)
#         k_chunk = ks[:, :, k, :, :].clone()
#         v_chunk = vs[:, :, k, :, :].clone()
#         q_chunk = qs[:, :, k, :, :].clone()
#         d_vec_chunk = d_vec[:, :, k, :, :].clone()
#         k_chunk_gated = k_chunk * torch.exp(d_vec_chunk - m_state)

#         C_state = (
#             torch.exp(g_state + m_prev_state - m_state) * C_prev_state
#             + k_chunk_gated.transpose(-2, -1) @ v_chunk
#         )
#         C_states[:, :, k] = C_state

#         n_state = torch.exp(
#             g_state + m_prev_state - m_state
#         ) * n_prev_state + k_chunk_gated.transpose(-2, -1).sum(-1)
#         n_states[:, :, k] = n_state

#         H_states[:, :, k, :, :] = (
#             q_chunk
#             @ C_state
#             / torch.max(torch.abs(q_chunk @ n_state.unsqueeze(-1)), m_state)
#         )
#         #     g_vec = torch.cat([g_vec, log_fgates[:, :, i].sum(-1)], dim=-1)

#         m_prev_state = m_state
#         C_prev_state = C_state
#         n_prev_state = n_state

#     # H_y = rearrange(H_states, "b nh nc l dh -> b nh (nc l) dh")
#     return H_states


# #! Not working,
# # should implement the materialize version of the GLA
# # second loop still missing
# def vlstm_chunkwise_parallel_2(
#     queries: torch.Tensor,
#     keys: torch.Tensor,
#     values: torch.Tensor,
#     igate_preact: torch.Tensor,
#     fgate_preact: torch.Tensor,
#     chunk_size: int = 64,
# ):
#     B, NH, S, DH = queries.shape
#     _dtype, _device = queries.dtype, queries.device
#     qs = rearrange(queries, "b nh (nc l) dh -> b nh nc l dh", l=chunk_size) * (DH**-0.5)
#     ks = rearrange(keys, "b nh (nc l) dh -> b nh nc l dh", l=chunk_size)
#     vs = rearrange(values, "b nh (nc l) dh -> b nh nc l dh", l=chunk_size)
#     _, _, NC, L, _ = qs.shape
#     igs = rearrange(igate_preact, "b nh (nc l) 1 -> b nh nc l", l=chunk_size)
#     fgs = rearrange(fgate_preact, "b nh (nc l) 1 -> b nh nc l", l=chunk_size)
#     log_fgates = F.logsigmoid(fgs)
#     # print(ks.shape)
#     # print(log_fgates.shape)

#     g_vec = log_fgates.sum(-1)
#     print(g_vec.shape)

#     p_vec_f = torch.cat(
#         [
#             torch.zeros((B, NH, NC, 1), dtype=_dtype, device=_device),
#             log_fgates[:, :, :, :-1].cumsum(-1),
#         ],
#         dim=-1,
#     )
#     print("p_vec: ", p_vec_f.shape)
#     print(p_vec_f)
#     print(log_fgates)

#     q_vec_f_raw = torch.cat(
#         [
#             torch.zeros((B, NH, NC, 1), dtype=_dtype, device=_device),
#             log_fgates[:, :, :, 1:].cumsum(-1),
#         ],
#         dim=-1,
#     )
#     q_vec_f = log_fgates[:, :, :, 1:].sum(-1, keepdim=True) - q_vec_f_raw
#     print("q_vec: ", q_vec_f.shape)
#     print(q_vec_f)
#     p_vec = p_vec_f + igs
#     q_vec = q_vec_f + igs
#     print("p_vec: ", p_vec.shape)
#     print("q_vec: ", q_vec.shape)

#     m_k = torch.zeros((B, NH, 1), dtype=_dtype, device=_device)
#     m_prev_k = torch.zeros((B, NH, 1), dtype=_dtype, device=_device)
#     C_k = torch.zeros((B, NH, DH, DH), dtype=_dtype, device=_device)
#     C_prev_k = torch.zeros((B, NH, DH, DH), dtype=_dtype, device=_device)
#     n_k = torch.zeros((B, NH, DH), dtype=_dtype, device=_device)
#     n_prev_k = torch.zeros((B, NH, DH), dtype=_dtype, device=_device)

#     # get the maximum values per chunk for p and q
#     p_vec_max = p_vec.max(-1).values
#     q_vec_max = q_vec.max(-1).values

#     C_states = torch.zeros((B, NH, NC, DH, DH), dtype=_dtype, device=_device)
#     n_states = torch.zeros((B, NH, NC, DH), dtype=_dtype, device=_device)
#     H_states = torch.zeros((B, NH, NC, L, DH), dtype=_dtype, device=_device)

#     # TODO the indices to not match yet use different q k v idx (see gla paper)
#     # + the intra chunk at k is missing
#     for k in range(1, NC):
#         i = k - 1
#         m_q_k = q_vec_max[:, :, i]
#         g_k = g_vec[:, :, i]
#         m_k = torch.max(g_k + m_prev_k, m_q_k)
#         k_chunk = ks[:, :, i, :, :].clone()
#         v_chunk = vs[:, :, i, :, :].clone()
#         q_chunk = qs[:, :, i, :, :].clone()
#         p_k = p_vec[:, :, i, :].clone()
#         q_k = q_vec[:, :, i, :].clone()
#         k_chunk_gated = k_chunk * torch.exp(q_k - m_k).unsqueeze(-1)

#         C_k = (
#             torch.exp(g_k + m_prev_k - m_k) * C_prev_k
#             + k_chunk_gated.transpose(-2, -1) @ v_chunk
#         )
#         C_states[:, :, k] = C_k

#         n_k = torch.exp(g_k + m_prev_k - m_k) * n_prev_k + k_chunk_gated.transpose(
#             -2, -1
#         ).sum(-1)
#         n_states[:, :, k] = n_k

#         m_p_k = p_vec_max[:, :, i]
#         m_H = torch.max(m_p_k, m_k)
#         q_chunk_gated = q_chunk * torch.exp(p_k - m_H).unsqueeze(-1)

#         H_states[:, :, i, :, :] = (
#             q_chunk_gated
#             @ C_prev_k
#             / torch.max(
#                 torch.abs(q_chunk_gated @ n_prev_k.unsqueeze(-1)), torch.exp(-m_k - m_H)
#             )
#         )
#         #     g_vec = torch.cat([g_vec, log_fgates[:, :, i].sum(-1)], dim=-1)

#         m_prev_k = m_k
#         C_prev_k = C_k
#         n_prev_k = n_k

#     return H_states


#! Somewhat working -> Inter chunk likely buggy
def vlstm_chunkwise_parallel_3(
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
    print(f"log_fgates: {log_fgates.shape}\n{log_fgates}")

    p_vec_f = torch.cat(
        [
            torch.zeros((B, NH, NC, 1), dtype=_dtype, device=_device),
            log_fgates[:, :, :, :-1].cumsum(-1),
        ],
        dim=-1,
    )
    p_vec_f = log_fgates[:, :, :, :].cumsum(-1)
    print(f"p_vec_f: {p_vec_f.shape}\n{p_vec_f}")

    q_vec_f_raw = torch.cat(
        [
            torch.zeros((B, NH, NC, 1), dtype=_dtype, device=_device),
            log_fgates[:, :, :, :-1].cumsum(-1),
        ],
        dim=-1,
    )
    q_vec_f = log_fgates[:, :, :, :].sum(-1, keepdim=True) - q_vec_f_raw
    print(f"q_vec_f: {q_vec_f.shape}\n{q_vec_f}")

    p_vec = p_vec_f + igs
    q_vec = q_vec_f + igs
    g_vec = log_fgates.sum(-1)
    print(f"g_vec: {g_vec.shape}\n{g_vec}")

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

        # ? Compute inter chunk contribution: H_inter
        p_k = p_vec[:, :, i, :].clone()

        m_p_k = p_vec_max[:, :, i]
        m_H = torch.max(m_p_k, m_k)
        q_chunk_gated = q_chunk * torch.exp(p_k - m_H).unsqueeze(-1)

        H_inter = (
            q_chunk_gated
            @ C_k
            / torch.max(
                torch.abs(q_chunk_gated @ n_k_inter.unsqueeze(-1)),
                torch.exp(-m_k - m_H),
            )
        )

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

        log_D_k = log_fg_k_matrix + log_ig_k.transpose(-2, -1)  # (B, NH, L, L)

        # compute the max state (for now isolated for intra chunk contribution)
        m_log_D_k = torch.max(log_D_k, dim=-1, keepdim=True).values

        log_D_k_stabilized = log_D_k - m_log_D_k
        D_k = torch.exp(log_D_k_stabilized)
        qk_k_matrix = q_chunk @ k_chunk.transpose(-2, -1)
        C_k_matrix = qk_k_matrix * D_k

        n_k_intra = torch.maximum(
            C_k_matrix.sum(dim=-1, keepdim=True).abs(), torch.exp(-m_log_D_k)
        )
        C_k_matrix_normalized = C_k_matrix / n_k_intra  # TODO add eps

        H_intra = C_k_matrix_normalized @ v_chunk  # (B, NH, L, DH)
        H_states[:, :, i, :, :] = H_inter + H_intra

    return H_states
