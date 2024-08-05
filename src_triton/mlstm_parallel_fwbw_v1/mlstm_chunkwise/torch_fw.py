# Copyright JKU Linz 2024
# Author: Maximilian Beck

import math

import torch
import torch.nn.functional as F
from einops import rearrange

"""PyTorch

mLSTM chunkwise parallel. 

Different versions of mLSTM chunkwise parallel forward pass.

Notation:
Dimensions:
    B: batch size
    NH: number of heads
    S: sequence length
    DH: hidden dimension
    NC: number of chunks
    L: chunk size

Variables:
    vecA, a: forward gate contribution, contribution of forget gates from last chunk state C_{k-1} to current timestep t
    vecB, b: backward gate contribution, contribution of forget and input gates up to next chunk state C_k (form current timestep t)
    vecG, g: "go through" gate contribution, contribution of forget gates from C_{k-1} to C_k.

"""


def mlstm_chunkwise_parallel_fw_looped(
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
    vecI: torch.Tensor,
    vecF: torch.Tensor,
    seq_chunk_size: int = 64,
):
    B, NH, S, DH = matQ.shape
    _dtype, _device = matQ.dtype, matQ.device
    matQ = rearrange(matQ, "b nh (nc l) dh -> b nh nc l dh", l=seq_chunk_size) * (
        DH**-0.5
    )
    matK = rearrange(matK, "b nh (nc l) dh -> b nh nc l dh", l=seq_chunk_size)
    matV = rearrange(matV, "b nh (nc l) dh -> b nh nc l dh", l=seq_chunk_size)
    _, _, NC, L, _ = matQ.shape
    vecI = rearrange(vecI, "b nh (nc l) -> b nh nc l", l=seq_chunk_size)
    vecF = rearrange(vecF, "b nh (nc l) -> b nh nc l", l=seq_chunk_size)

    # compute the gates, the g and the a and b vectors
    vecF_logsig = F.logsigmoid(vecF)

    vecB_fcs = vecF_logsig.cumsum(-1)
    vecA_frcs = vecF_logsig.sum(-1, keepdim=True) - vecB_fcs

    vecB = vecB_fcs
    vecA = vecA_frcs + vecI
    vecG = vecF_logsig.sum(-1)

    # get the maximum values per chunk for p and q
    vecB_max = vecB.max(-1).values
    vecA_max = vecA.max(-1).values

    # loop 1: materialize the  C_k, n_k, m_k
    matC_k_states = torch.zeros((B, NH, NC, DH, DH), dtype=_dtype, device=_device)
    vecN_k_states = torch.zeros((B, NH, NC, DH), dtype=_dtype, device=_device)
    scaMinter_k_states = torch.zeros((B, NH, NC, 1), dtype=_dtype, device=_device)

    m_k_inter = torch.zeros((B, NH, 1), dtype=_dtype, device=_device)
    m_prev_k = torch.zeros((B, NH, 1), dtype=_dtype, device=_device)
    C_k = torch.zeros((B, NH, DH, DH), dtype=_dtype, device=_device)
    C_prev_k = torch.zeros((B, NH, DH, DH), dtype=_dtype, device=_device)
    n_k = torch.zeros((B, NH, DH), dtype=_dtype, device=_device)
    n_prev_k = torch.zeros((B, NH, DH), dtype=_dtype, device=_device)
    for k in range(1, NC):

        # m_k
        m_a_k = vecA_max[:, :, k - 1]
        g_k = vecG[:, :, k - 1]
        m_k_inter = torch.max(g_k + m_prev_k, m_a_k)
        scaMinter_k_states[:, :, k] = m_k_inter

        # C_k
        matK_chunk = matK[:, :, k - 1, :, :].clone()
        matV_chunk = matV[:, :, k - 1, :, :].clone()
        a_k = vecA[:, :, k - 1, :].clone()

        matK_chunk_gated = matK_chunk * torch.exp(a_k - m_k_inter).unsqueeze(-1)

        C_k = (
            torch.exp(g_k + m_prev_k - m_k_inter) * C_prev_k
            + matK_chunk_gated.transpose(-2, -1) @ matV_chunk
        )
        matC_k_states[:, :, k] = C_k

        # n_k
        n_k = torch.exp(
            g_k + m_prev_k - m_k_inter
        ) * n_prev_k + matK_chunk_gated.transpose(-2, -1).sum(-1)
        vecN_k_states[:, :, k] = n_k

        # move to the next iteration
        m_prev_k = m_k_inter
        C_prev_k = C_k
        n_prev_k = n_k

    ltr = torch.tril(
        torch.ones(
            (L, L),
            dtype=torch.bool,
            device=_device,
        )
    )

    # loop 2: compute the H_states
    H_states = torch.zeros((B, NH, NC, L, DH), dtype=_dtype, device=_device)
    for k in range(1, NC + 1):

        # load C_k, n_k, m_k
        C_k = matC_k_states[:, :, k - 1]
        n_k_inter = vecN_k_states[:, :, k - 1]
        m_k_inter = scaMinter_k_states[:, :, k - 1]

        # load q, k, v chunks
        matQ_chunk = matQ[:, :, k - 1, :, :].clone()
        matK_chunk = matK[:, :, k - 1, :, :].clone()
        matV_chunk = matV[:, :, k - 1, :, :].clone()

        # ? Compute intra chunk contribution: H_intra
        vecF_logsig_chunk = vecF_logsig[:, :, k - 1]
        vecF_logsig_cs_chunk = vecF_logsig_chunk.cumsum(-1)

        vecI_chunk = vecI[:, :, k - 1]

        matF_logsig_chunk = (
            vecF_logsig_cs_chunk[:, :, :, None] - vecF_logsig_cs_chunk[:, :, None, :]
        )

        matF_logsig_mask_chunk = torch.where(ltr, matF_logsig_chunk, -float("inf"))

        matLogD_chunk = matF_logsig_mask_chunk + vecI_chunk[:, :, None, :]

        # max_state intra
        vecMintra_k = torch.max(
            matLogD_chunk, dim=-1, keepdim=False
        ).values  # (B, NH, L)

        # max_state inter
        vecB_k_chunk = vecB[:, :, k - 1]  # (B, NH, L)

        # max_state combined
        vecM_b_inter = vecB_k_chunk + m_k_inter
        vecM_k_inter_intra = torch.maximum(vecM_b_inter, vecMintra_k)  # (B, NH, L)

        vecM_k_inter_intra = vecM_k_inter_intra[:, :, :, None]  # (B, NH, L, 1)
        vecM_b_inter = vecM_b_inter[:, :, :, None]  # (B, NH, L, 1)

        matLogD_stabilized_chunk = matLogD_chunk - vecM_k_inter_intra
        matD_chunk = torch.exp(matLogD_stabilized_chunk)

        # matQ_chunk is alraedy scaled by DH**-0.5
        matS_chunk = matQ_chunk @ matK_chunk.transpose(-2, -1)

        matM_chunk = matS_chunk * matD_chunk

        # ? Combine H_intra with H_inter
        matQ_chunk_gated = matQ_chunk * torch.exp(vecM_b_inter - vecM_k_inter_intra)

        numerator_common = matQ_chunk_gated @ C_k + matM_chunk @ matV_chunk

        denom_common = matQ_chunk_gated @ n_k_inter.unsqueeze(-1) + matM_chunk.sum(
            dim=-1, keepdim=True
        )

        matH_k_chunk = numerator_common / torch.maximum(
            torch.abs(denom_common), torch.exp(-vecM_k_inter_intra)
        )

        H_states[:, :, k - 1, :, :] = matH_k_chunk

    H_out = rearrange(H_states, "b nh nc l dh -> b nh (nc l) dh")
    return H_out


def mlstm_chunkwise_parallel_fw_parallel(
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
    vecI: torch.Tensor,
    vecF: torch.Tensor,
    seq_chunk_size: int = 64,
    detach_denominator: bool = False,
):
    B, NH, S, DH = matQ.shape
    _dtype, _device = matQ.dtype, matQ.device
    matQ = rearrange(matQ, "b nh (nc l) dh -> b nh nc l dh", l=seq_chunk_size) * (
        DH**-0.5
    )
    matK = rearrange(matK, "b nh (nc l) dh -> b nh nc l dh", l=seq_chunk_size)
    matV = rearrange(matV, "b nh (nc l) dh -> b nh nc l dh", l=seq_chunk_size)
    _, _, NC, L, _ = matQ.shape
    vecI = rearrange(vecI, "b nh (nc l) -> b nh nc l", l=seq_chunk_size)
    vecF = rearrange(vecF, "b nh (nc l) -> b nh nc l", l=seq_chunk_size)

    # compute the gates, the g and the a and b vectors
    vecF_logsig = F.logsigmoid(vecF)

    vecB_fcs = vecF_logsig.cumsum(-1)
    vecA_frcs = vecF_logsig.sum(-1, keepdim=True) - vecB_fcs

    vecB = vecB_fcs
    vecA = vecA_frcs + vecI
    vecG = vecF_logsig.sum(-1)

    # get the maximum values per chunk for p and q
    vecB_max = vecB.max(-1).values
    vecA_max = vecA.max(-1).values

    #! loop 1: materialize the  C_k, n_k, m_k
    matC_k_states = torch.zeros((B, NH, NC, DH, DH), dtype=_dtype, device=_device)
    vecN_k_states = torch.zeros((B, NH, NC, DH), dtype=_dtype, device=_device)
    scaMinter_k_states = torch.zeros((B, NH, NC, 1), dtype=_dtype, device=_device)

    # initial states
    m_prev_k = torch.zeros((B, NH, 1), dtype=_dtype, device=_device)
    C_prev_k = torch.zeros((B, NH, DH, DH), dtype=_dtype, device=_device)
    n_prev_k = torch.zeros((B, NH, DH), dtype=_dtype, device=_device)

    # m_k_inter = torch.zeros((B, NH, 1), dtype=_dtype, device=_device)
    # C_k = torch.zeros((B, NH, DH, DH), dtype=_dtype, device=_device)
    # n_k = torch.zeros((B, NH, DH), dtype=_dtype, device=_device)
    for k in range(1, NC):

        # m_k
        m_a_k = vecA_max[:, :, k - 1]
        g_k = vecG[:, :, k - 1]
        m_k_inter = torch.max(g_k + m_prev_k, m_a_k)
        scaMinter_k_states[:, :, k] = m_k_inter

        # C_k
        matK_chunk = matK[:, :, k - 1, :, :].clone()
        matV_chunk = matV[:, :, k - 1, :, :].clone()
        a_k = vecA[:, :, k - 1, :].clone()

        matK_chunk_gated = matK_chunk * torch.exp(a_k - m_k_inter).unsqueeze(-1)

        C_k = (
            torch.exp(g_k + m_prev_k - m_k_inter) * C_prev_k
            + matK_chunk_gated.transpose(-2, -1) @ matV_chunk
        )
        matC_k_states[:, :, k] = C_k

        # n_k
        n_k = torch.exp(
            g_k + m_prev_k - m_k_inter
        ) * n_prev_k + matK_chunk_gated.transpose(-2, -1).sum(-1)
        vecN_k_states[:, :, k] = n_k

        # move to the next iteration
        m_prev_k = m_k_inter
        C_prev_k = C_k
        n_prev_k = n_k

    ltr = torch.tril(
        torch.ones(
            (L, L),
            dtype=torch.bool,
            device=_device,
        )
    )

    #! compute the H_states in parallel

    # ? Compute intra chunk contribution: H_intra
    vecF_logsig_cs_chunk = vecF_logsig.cumsum(-1)

    matF_logsig_chunk = (
        vecF_logsig_cs_chunk[:, :, :, :, None] - vecF_logsig_cs_chunk[:, :, :, None, :]
    )

    matF_logsig_mask_chunk = torch.where(ltr, matF_logsig_chunk, -float("inf"))

    matLogD_chunk = matF_logsig_mask_chunk + vecI[:, :, :, None, :]

    # max_state intra
    vecMintra_k = torch.max(
        matLogD_chunk, dim=-1, keepdim=False
    ).values  # (B, NH, NC, L)

    # max_state combined
    vecM_b_inter = vecB + scaMinter_k_states  # (B, NH, NC, L)
    vecM_k_inter_intra = torch.maximum(vecM_b_inter, vecMintra_k)  # (B, NH, NC, L)

    vecM_k_inter_intra = vecM_k_inter_intra[:, :, :, :, None]  # (B, NH, NC, L, 1)
    vecM_b_inter = vecM_b_inter[:, :, :, :, None]  # (B, NH, NC, L, 1)

    matLogD_stabilized_chunk = matLogD_chunk - vecM_k_inter_intra
    matD_chunk = torch.exp(matLogD_stabilized_chunk)

    # matQ_chunk is alraedy scaled by DH**-0.5
    matS_chunk = matQ @ matK.transpose(-2, -1)

    matM_chunk = matS_chunk * matD_chunk

    # ? Combine H_intra with H_inter
    matQ_chunk_gated = matQ * torch.exp(vecM_b_inter - vecM_k_inter_intra)

    numerator_common = matQ_chunk_gated @ matC_k_states + matM_chunk @ matV

    denom_common = matQ_chunk_gated @ vecN_k_states.unsqueeze(-1) + matM_chunk.sum(
        dim=-1, keepdim=True
    )

    max_denom_common = torch.maximum(
        torch.abs(denom_common), torch.exp(-vecM_k_inter_intra)
    )

    if detach_denominator:
        max_denom_common = max_denom_common.detach()

    matH_k_chunk = numerator_common / max_denom_common

    H_out = rearrange(matH_k_chunk, "b nh nc l dh -> b nh (nc l) dh")
    return H_out
