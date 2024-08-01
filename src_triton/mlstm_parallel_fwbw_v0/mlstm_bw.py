import math

import torch
import triton
import triton.language as tl

"""
In this file:
- parallel backward pass of the mlstm
"""


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton.Config({"BLOCK_Q": BQ, "BLOCK_KV": BKV}, num_stages=s, num_warps=w)
    for BQ, BKV in [
        (128, 128),
        (128, 64),
        (128, 32),
        (128, 16),
        (64, 64),
        (64, 32),
        (64, 16),
        (32, 32),
        (32, 16),
        (16, 16),
    ]
    for s in [3, 4, 7]
    for w in [4, 8]
]


def keep(conf):
    # filter out configurations that are smaller than (128, 128) when using 8 warps
    BLOCK_M = conf.kwargs["BLOCK_Q"]
    BLOCK_N = conf.kwargs["BLOCK_KV"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


BLOCK_Q = 16
BLOCK_KV = 16


# @triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _mlstm_bwd(
    matDeltaHtilde,
    matQ,
    matK,
    matV,
    vecI,
    vecF_cs,
    vecM,
    vecN,
    qk_scale,
    matDeltaQ,
    matDeltaK,
    matDeltaV,
    vecDeltaI,
    stride_dhtz,
    stride_dhth,
    stride_dhts,
    stride_dhtd,
    stride_qz,
    stride_qh,
    stride_qs,
    stride_qd,  #
    stride_kz,
    stride_kh,
    stride_ks,
    stride_kd,  #
    stride_vz,
    stride_vh,
    stride_vs,
    stride_vd,  #
    stride_ifmn_z,
    stride_ifmn_h,
    stride_ifmn_s,
    Z,
    H,
    N_CTX,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_Q: tl.constexpr,  #
    BLOCK_KV: tl.constexpr,  #
    EPS: tl.constexpr = 1e-6,
):
    ## Notation
    # z: batch size
    # h: number of heads
    # s: sequence length
    # d: head dimension

    tl.static_assert(BLOCK_KV <= HEAD_DIM)
    kvIdx = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    qkvh_batchhead_offset = (
        off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    )
    ifmn_batchhead_offset = (
        off_z.to(tl.int64) * stride_ifmn_z + off_h.to(tl.int64) * stride_ifmn_h
    )

    # input block pointers
    # Note: the order argument specifies the memory traversal order within a block
    matDeltaHtilde_block_ptr = tl.make_block_ptr(
        base=matDeltaHtilde + qkvh_batchhead_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_dhts, stride_dhtd),
        offsets=(0, 0),
        block_shape=(BLOCK_Q, HEAD_DIM),
        order=(1, 0),
    )
    matQ_block_ptr = tl.make_block_ptr(
        base=matQ + qkvh_batchhead_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qs, stride_qd),
        offsets=(0, 0),
        block_shape=(BLOCK_Q, HEAD_DIM),
        order=(1, 0),
    )
    matK_block_ptr = tl.make_block_ptr(
        base=matK + qkvh_batchhead_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_ks, stride_kd),
        offsets=(kvIdx * BLOCK_KV, 0),
        block_shape=(BLOCK_KV, HEAD_DIM),
        order=(1, 0),
    )
    matV_block_ptr = tl.make_block_ptr(
        base=matV + qkvh_batchhead_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vs, stride_vd),
        offsets=(kvIdx * BLOCK_KV, 0),
        block_shape=(BLOCK_KV, HEAD_DIM),
        order=(1, 0),
    )

    # output block pointers
    matDeltaQ_block_ptr = tl.make_block_ptr(
        base=matDeltaQ + qkvh_batchhead_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_dhts, stride_dhtd),
        offsets=(0, 0),
        block_shape=(BLOCK_Q, HEAD_DIM),
        order=(1, 0),
    )
    matDeltaK_block_ptr = tl.make_block_ptr(
        base=matDeltaK + qkvh_batchhead_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_dhts, stride_dhtd),
        offsets=(0, 0),
        block_shape=(BLOCK_KV, HEAD_DIM),
        order=(1, 0),
    )
    matDeltaV_block_ptr = tl.make_block_ptr(
        base=matDeltaV + qkvh_batchhead_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_dhts, stride_dhtd),
        offsets=(0, 0),
        block_shape=(BLOCK_KV, HEAD_DIM),
        order=(1, 0),
    )

    # ? LOADING AND INITIALIZATION
    # load matK_tile, matV_tile (will stay in SRAM throughout)
    # TODO
    # init matDeltaK_tile, matDeltaV_tile accumulators
    # TODO
    # load vecI_chunk
    # TODO
    # init vecDeltaI_sum accumulator
    # TODO
    # load vecF_cs_chunk_KV
    # ifmn_offset defines the proper batch-head, q_offset defines the location
    # in the sequence for the current thread block
    vecF_cs_chunk_KV_ptr = (
        vecF_cs + ifmn_batchhead_offset + kv_offset + tl.arange(0, BLOCK_KV)
    )
    vecF_cs_chunk_KV = tl.load(vecF_cs_chunk_KV_ptr)
    vecF_cs_chunk_KV = vecF_cs_chunk_KV.to(tl.float32)

    # define kv_block_idxes for causal masking
    kv_offset = kvIdx * BLOCK_KV
    kv_block_idxes = kv_offset + tl.arange(0, BLOCK_Q)

    # ? MAIN LOOP
    qStartIdx = (kvIdx * BLOCK_KV) // BLOCK_Q
    qEndIdx = tl.cdiv(N_CTX, BLOCK_Q)
    qEndIdx = tl.multiple_of(qEndIdx, BLOCK_Q)

    matQ_block_ptr = ...  # TODO: tl.advance(matQ_block_ptr, (0, lo))
    matDeltaQ_block_ptr = ...  # TODO: tl.advance(matV_block_ptr, (lo, 0))

    # loop over BLOCK_Q dimension and update matDeltK, matDeltaV, vecDeltaI_sum accumulators
    # update matDeltaQ synchronously across all thread blocks (atomic add in HBM)
    for qIdx in range(qStartIdx, qEndIdx):
        q_offset = qIdx * BLOCK_Q
        q_offset = tl.multiple_of(q_offset, BLOCK_Q)

        # TODO from here
        # -- compute matS --
        k = tl.load(matK_block_ptr)
        matS = tl.dot(q, k)
        matS = matS / qk_scale

        # ? -- create gate matrix tile D --
        # load vecF_cs_chunkKV
        vecF_cs_vecI_chunkKV_offset = (
            ifmn_batchhead_offset + start_n + tl.arange(0, BLOCK_KV)
        )
        vecF_cs_chunkKV_ptr = vecF_cs + vecF_cs_vecI_chunkKV_offset
        vecF_cs_chunkKV = tl.load(vecF_cs_chunkKV_ptr)
        vecF_cs_chunkKV = vecF_cs_chunkKV.to(tl.float32)

        # load vecI_chunkKV
        vecI_ptr = vecI + vecF_cs_vecI_chunkKV_offset
        vecI_chunkKV = tl.load(vecI_ptr)
        vecI_chunkKV = vecI_chunkKV.to(tl.float32)

        # compute D matrix
        matD_log_fgates = vecF_cs_chunk_KV[:, None] - vecF_cs_chunkKV[None, :]
        matD = matD_log_fgates + vecI_chunkKV[None, :]

        # ? -- causal masking --
        #! TODO with this if I get a weird error: operation scheduled before its operands
        if start_n >= kv_offset:
            # we are on diagonal
            kv_block_idxes = start_n + tl.arange(0, BLOCK_KV)
            mask = q_block_idxes[:, None] - kv_block_idxes[None, :]
            matD = tl.where(mask >= 0, matD, -float("inf"))

        # else: below diagonal

        # ? -- compute m_state --
        m_temp = tl.max(matD, axis=1)  # rowwise max
        m_temp = tl.maximum(MINIMUM_MAX_VAL, m_temp)  # elementwise max
        m_new = tl.maximum(m_old, m_temp)
        m_ratio = tl.exp(m_old - m_new)

        # ? -- compute matC --
        matD_tilde = tl.exp(matD - m_new[:, None])
        matC = matS * matD_tilde

        # ? -- compute l_state --
        # tl.fma did not bring performance improvement
        l_temp = m_ratio * l_old
        l_new = l_temp + tl.sum(matC, axis=1)

        # ? -- compute n_state --
        n_new = tl.maximum(tl.abs(l_new), tl.exp(-m_new))

        # ? -- compute h_out -- update h_out --
        # compute weighting factor
        # tl.fdiv did not bring any performance improvement
        h_out_old_weight = (m_ratio * n_old) / n_new
        h_out = h_out * h_out_old_weight[:, None]

        v = tl.load(matV_block_ptr)

        matC = matC / n_new[:, None]
        matC = matC.to(tl.float16)
        h_out = tl.dot(matC, v, h_out)

        matV_block_ptr = tl.advance(matV_block_ptr, (BLOCK_KV, 0))
        matK_block_ptr = tl.advance(matK_block_ptr, (0, BLOCK_KV))

        l_old = l_new
        m_old = m_new
        n_old = n_new

    # epilogue
    tl.store(H_block_ptr, h_out.to(matH.type.element_ty))
    vecM_ptr = vecM + ifmn_batchhead_offset + kv_offset + tl.arange(0, BLOCK_Q)
    vecN_ptr = vecN + ifmn_batchhead_offset + kv_offset + tl.arange(0, BLOCK_Q)
    tl.store(vecM_ptr, m_old.to(vecM.type.element_ty))
    tl.store(vecN_ptr, n_old.to(vecN.type.element_ty))


def mlstm_bw(
    matDeltaHtilde: torch.Tensor,
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
    vecI: torch.Tensor,
    vecF: torch.Tensor,
    vecM: torch.Tensor,
    vecN: torch.Tensor,
    BLOCK_Q: int = BLOCK_Q,
    BLOCK_KV: int = BLOCK_KV,
) -> torch.Tensor:
    # batch size, number of heads, sequence length, head dimension
    BS, NH, SL, DH = matQ.shape
    assert vecI.shape == (BS, NH, SL)
    assert vecF.shape == (BS, NH, SL)
    assert vecM.shape == (BS, NH, SL)
    assert vecN.shape == (BS, NH, SL)
    assert matDeltaHtilde.shape == (BS, NH, SL, DH)
    assert matQ.shape == matK.shape == matV.shape

    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = matQ.shape[-1], matK.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    HEAD_DIM_V = matV.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    # assert HEAD_DIM_K in {16, 32, 64, 128, 256}

    # grid = lambda args: (
    #     triton.cdiv(SL, args["BLOCK_KV"]),
    #     BS * NH,
    #     1,
    # )
    # fix grid for debugging
    grid = lambda args: (
        triton.cdiv(SL, BLOCK_KV),
        BS * NH,
        1,
    )
    print(f"Triton grid: {grid(None)}, BLOCK_Q: {BLOCK_Q}, BLOCK_KV: {BLOCK_KV}")

    ## ? preprocessing, initialization
    matDeltaQ = torch.empty_like(matQ)
    matDeltaK = torch.empty_like(matK)
    matDeltaV = torch.empty_like(matV)

    vecDeltaI = torch.zeros_like(vecI)

    vecF_cs = torch.nn.functional.logsigmoid(vecF).cumsum(-1)
    ## ? end preprocessing

    # strides for matQ, matK, matV are same as matDeltaQ, matDeltaK, matDeltaV
    _mlstm_bwd[grid](
        matDeltaHtilde=matDeltaHtilde.contiguous(),
        matQ=matQ.contiguous(),
        matK=matK.contiguous(),
        matV=matV.contiguous(),
        vecI=vecI.contiguous(),
        vecF_cs=vecF_cs.contiguous(),
        vecM=vecM.contiguous(),
        vecN=vecN.contiguous(),
        qk_scale=math.sqrt(HEAD_DIM_Q),
        matDeltaQ=matDeltaQ,
        matDeltaK=matDeltaK,
        matDeltaV=matDeltaV,
        vecDeltaI=vecDeltaI,
        stride_qz=matQ.stride(0),
        stride_qh=matQ.stride(1),
        stride_qs=matQ.stride(2),
        stride_qd=matQ.stride(3),
        stride_kz=matK.stride(0),
        stride_kh=matK.stride(1),
        stride_ks=matK.stride(2),
        stride_kd=matK.stride(3),
        stride_vz=matV.stride(0),
        stride_vh=matV.stride(1),
        stride_vs=matV.stride(2),
        stride_vd=matV.stride(3),
        stride_ifmn_z=vecF_cs.stride(0),
        stride_ifmn_h=vecF_cs.stride(1),
        stride_ifmn_s=vecF_cs.stride(2),
        Z=BS,
        H=NH,
        N_CTX=SL,
        HEAD_DIM=HEAD_DIM_K,
        BLOCK_Q=BLOCK_Q,
        BLOCK_KV=BLOCK_KV,
    )

    ## ? postprocessing
    # compute the vecDeltaFbar values with dfbar = rev_cumsum((q*dq - k*dk).sum(-1))
    vecDeltaFbar_acc = (matQ * matDeltaQ - matK * matDeltaK).sum(-1)
    vecDeltaFbar = vecDeltaFbar_acc.flip(-1).cumsum(-1).flip(-1)
    vecDeltaF = vecDeltaFbar * torch.sigmoid(-vecF)
    ## ? end postprocessing

    return matDeltaQ, matDeltaK, matDeltaV, vecDeltaI, vecDeltaF
