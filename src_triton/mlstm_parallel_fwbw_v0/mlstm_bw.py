import math

import torch
import triton
import triton.language as tl

"""
In this file:
- parallel backward pass of the mlstm

Note: 
- There is a difference in the way the deltaQ gradient is computed in the backward pass for 
  the triton implementation of the official triton tutorial implementation and the triton implementation 
  in the flash attention repo.
- The difference is:
    - triton tutorial impl: There are two loops (functions). 
      The first implements the dk and dv gradients. The second implements the dq gradient. 
      Advantage: we do not have a race condition on the dq gradient in global memory (CUDA impl solves this with atomic add.)
      Disadvantage: we basically compute the Attention matrix twice.
    - flash attention impl: There is only one loop and the dq gradient is computed in the same loop as the dk and dv gradients.
      Advantage: we compute the Attention matrix only once.
      Disadvantage: there might be a race condition on the dq gradient in global memory as all threadblocks need to update the same dq gradient
                    memory space.
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
    # directly transpose matDeltaV while loading
    matDeltaV_block_ptr = tl.make_block_ptr(
        base=matDeltaV + qkvh_batchhead_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_dhts, stride_dhtd),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_KV),  # this is the transposed shape in SRAM
        order=(
            0,
            1,
        ),  # adapt the order to the underlying layout (which is not transposed), we load HEAD_DIM first
    )

    # ? LOADING AND INITIALIZATION
    # define kv_block_idxes for causal masking
    kv_offset = kvIdx * BLOCK_KV
    kv_offset = tl.multiple_of(kv_offset, BLOCK_KV)
    kv_block_idxes = kv_offset + tl.arange(0, BLOCK_KV)

    # load matK_tile, matV_tile
    matK_tile = tl.load(matK_block_ptr)  # (BLOCK_KV, HEAD_DIM)
    matV_tile = tl.load(matV_block_ptr)  # (HEAD_DIM, BLOCK_KV)
    # init matDeltaK_tile, matDeltaV_tile accumulators
    matDeltaK_tile = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    matDeltaV_tile = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    # load vecI_chunk
    vecI_chunk_KV_ptr = (
        vecI + ifmn_batchhead_offset + kv_offset + tl.arange(0, BLOCK_KV)
    )
    vecI_chunk_KV = tl.load(vecI_chunk_KV_ptr)  # (BLOCK_KV,)

    # init vecDeltaI_sum accumulator
    vecDeltaI_sum_chunk_KV = tl.zeros([BLOCK_KV], dtype=tl.float32)

    # load vecF_cs_chunk_KV
    vecF_cs_chunk_KV_ptr = (
        vecF_cs + ifmn_batchhead_offset + kv_offset + tl.arange(0, BLOCK_KV)
    )
    vecF_cs_chunk_KV = tl.load(vecF_cs_chunk_KV_ptr)
    vecF_cs_chunk_KV = vecF_cs_chunk_KV.to(tl.float32)

    # ? MAIN LOOP
    qStartIdx = (kvIdx * BLOCK_KV) // BLOCK_Q
    qEndIdx = tl.cdiv(N_CTX, BLOCK_Q)
    qEndIdx = tl.multiple_of(qEndIdx, BLOCK_Q)

    # move mat(Delta)Q_block_ptr & matDeltaHtilde_block_ptr to the position for the current thread block
    # input pointers:
    matQ_block_ptr = tl.advance(matQ_block_ptr, (qStartIdx * BLOCK_Q, 0))
    matDeltaHtilde_block_ptr = tl.advance(
        matDeltaHtilde_block_ptr, (qStartIdx * BLOCK_Q, 0)
    )
    # output pointers:
    matDeltaQ_block_ptr = tl.advance(matDeltaQ_block_ptr, (qStartIdx * BLOCK_Q, 0))

    # loop over BLOCK_Q dimension and update matDeltK, matDeltaV, vecDeltaI_sum accumulators
    # update matDeltaQ synchronously across all thread blocks (atomic add in HBM)
    for qIdx in range(qStartIdx, qEndIdx):
        q_offset = qIdx * BLOCK_Q
        q_offset = tl.multiple_of(q_offset, BLOCK_Q)

        # load matQ_tile & matDeltaHtilde_tile
        matQ_tile = tl.load(matQ_block_ptr)  # (BLOCK_Q, HEAD_DIM)
        matDeltaHtilde_tile = tl.load(matDeltaHtilde_block_ptr)  # (BLOCK_Q, HEAD_DIM)

        # load vecM_chunk_Q, vecN_chunk_Q
        vecMN_offsets = ifmn_batchhead_offset + q_offset + tl.arange(0, BLOCK_Q)
        vecM_chunk_Q_ptr = vecM + vecMN_offsets
        vecN_chunk_Q_ptr = vecN + vecMN_offsets

        vecM_chunk_Q = tl.load(vecM_chunk_Q_ptr)  # (BLOCK_Q,)
        vecN_chunk_Q = tl.load(vecN_chunk_Q_ptr)  # (BLOCK_Q,)

        # load vecF_cs_chunk_Q
        vecF_cs_chunk_Q_ptr = (
            vecF_cs + ifmn_batchhead_offset + q_offset + tl.arange(0, BLOCK_Q)
        )
        vecF_cs_chunk_Q = tl.load(vecF_cs_chunk_Q_ptr)
        vecF_cs_chunk_Q = vecF_cs_chunk_Q.to(tl.float32)

        # compute matDeltaC_tile
        matDeltaC_tile = tl.dot(matDeltaHtilde_tile, matV_tile)  # (BLOCK_Q, BLOCK_KV)
        matDeltaC_tile = matDeltaC_tile / (vecN_chunk_Q[:, None] + EPS)

        # ? recomputation of S & D matrices
        # compute matS_tile
        matK_tile_transposed = tl.trans(matK_tile, (1, 0))  # (HEAD_DIM, BLOCK_KV)
        matS_tile = torch.dot(matQ_tile, matK_tile_transposed)  # (BLOCK_Q, BLOCK_KV)
        matS_tile = matS_tile / qk_scale

        # compute matLogD_tile
        matLogD_Fgates_tile = vecF_cs_chunk_Q[:, None] - vecF_cs_chunk_KV[None, :]
        matLogD_tile = matLogD_Fgates_tile + vecI_chunk_KV[None, :]

        # causal masking
        if kv_offset >= q_offset:
            # we are on diagonal
            q_block_idxes = q_offset + tl.arange(0, BLOCK_Q)
            mask = q_block_idxes[:, None] - kv_block_idxes[None, :]
            # we set all values above the main diagonal to -inf
            matLogD_tile = tl.where(mask >= 0, matLogD_tile, -float("inf"))

        # else: below main diagonal

        matDprime_tile = tl.exp(
            matLogD_tile - vecM_chunk_Q[:, None]
        )  # (BLOCK_Q, BLOCK_KV)
        # ? end recomputation of S & D matrices

        matDeltaCTilde_tile = matDeltaC_tile * matS_tile * matDprime_tile

        # compute sum for vecDeltaI
        # sum up the columns of matDeltaCTilde_tile
        vecDeltaI_sum_chunk_KV += tl.sum(matDeltaCTilde_tile, axis=0)  # (BLOCK_KV,)

        matP_tile = matDeltaC_tile * matDprime_tile  # (BLOCK_Q, BLOCK_KV)
        matR_tile = matS_tile * matDprime_tile  # (BLOCK_Q, BLOCK_KV)

        # update matDeltaQ_tile in HBM
        matDeltaQ_tile = tl.dot(matP_tile, matK_tile)  # (BLOCK_Q, HEAD_DIM)
        matDeltaQ_tile = matDeltaQ_tile / qk_scale
        tl.atomic_add(matDeltaQ_block_ptr, matDeltaQ_tile)

        # update matDeltaK_tile, matDeltaV_tile in SRAM
        matP_tile_transposed = tl.trans(matP_tile, (1, 0))  # (BLOCK_KV, BLOCK_Q)
        matDeltaK_tile_temp = tl.dot(
            matP_tile_transposed, matQ_tile
        )  # (BLOCK_KV, HEAD_DIM)
        matDeltaK_tile += matDeltaK_tile_temp / qk_scale

        matR_tile_transposed = tl.trans(matR_tile, (1, 0))  # (BLOCK_KV, BLOCK_Q)
        matDeltaHtilde_tile_normalized = matDeltaHtilde_tile / (
            vecN_chunk_Q[:, None] + EPS
        )  # (BLOCK_Q, HEAD_DIM)
        matDeltaV_tile += tl.dot(
            matR_tile_transposed, matDeltaHtilde_tile_normalized
        )  # (BLOCK_KV, HEAD_DIM)

        # advance pointers (delta_)Q + deltaHtilde
        matQ_block_ptr = tl.advance(matQ_block_ptr, (BLOCK_Q, 0))
        matDeltaQ_block_ptr = tl.advance(matDeltaQ_block_ptr, (BLOCK_Q, 0))
        matDeltaHtilde_block_ptr = tl.advance(matDeltaHtilde_block_ptr, (BLOCK_Q, 0))

        # ? END MAIN LOOP

    # epilogue
    # store matDeltaK_tile, matDeltaV_tile
    tl.store(matDeltaK_block_ptr, matDeltaK_tile.to(matDeltaK.type.element_ty))
    tl.store(matDeltaV_block_ptr, matDeltaV_tile.to(matDeltaV.type.element_ty))
    # store vecDeltaI_sum
    vecDeltaI_chunk_KV_ptr = (
        vecDeltaI + ifmn_batchhead_offset + kv_offset + tl.arange(0, BLOCK_KV)
    )
    tl.store(
        vecDeltaI_chunk_KV_ptr, vecDeltaI_sum_chunk_KV.to(vecDeltaI.type.element_ty)
    )


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
