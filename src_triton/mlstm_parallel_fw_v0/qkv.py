import math

import torch
import triton
import triton.language as tl

"""
Goals: 
- Get to know triton
- implement the forward pass of the mlstm
"""


@triton.jit
def _qkv_fwd_inner(
    acc,
    l_i,
    m_i,
    q,  #
    K_block_ptr,
    V_block_ptr,  #
    start_m,
    qk_scale,  #
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
    N_CTX: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
    for BM in [64, 128]
    for BN in [32, 64]
    for s in [3, 4, 7]
    for w in [4, 8]
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


BLOCK_Q = 64
BLOCK_KV = 32


# @triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _qkv_fwd(
    matQ,
    matK,
    matV,
    qk_scale,
    matH,  #
    vecN,
    vecM,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  #
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,  #
    stride_hz,
    stride_hh,
    stride_hm,
    stride_hn,  #
    Z,
    H,
    N_CTX,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_Q: tl.constexpr,  #
    BLOCK_KV: tl.constexpr,  #
):
    tl.static_assert(BLOCK_KV <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=matQ + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, HEAD_DIM),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=matV + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_KV, HEAD_DIM),
        order=(0, 1),
    )
    K_block_ptr = tl.make_block_ptr(
        base=matK + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_KV),
        order=(0, 1),
    )
    H_block_ptr = tl.make_block_ptr(
        base=matH + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_hm, stride_hn),
        offsets=(start_m * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, HEAD_DIM),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_n = tl.arange(0, BLOCK_KV)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_Q], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    tl.static_print("q", q)
    k = tl.load(K_block_ptr)
    tl.static_print("k", k)
    v = tl.load(V_block_ptr)
    tl.static_print("v", v)
    tl.static_print("qk_scale", qk_scale)
    tl.device_print("qk_scale", qk_scale)

    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    # acc, l_i, m_i = _attn_fwd_inner(
    #     acc,
    #     l_i,
    #     m_i,
    #     q,
    #     K_block_ptr,
    #     V_block_ptr,  #
    #     start_m,
    #     qk_scale,  #
    #     BLOCK_Q,
    #     HEAD_DIM,
    #     BLOCK_KV,  #
    #     4 - STAGE,
    #     offs_m,
    #     offs_n,
    #     N_CTX,
    #     V.dtype.element_ty == tl.float8e5,  #
    # )
    # epilogue
    # m_i += tl.math.log2(l_i)
    # acc = acc / l_i[:, None]
    # m_ptrs = vecM + off_hz * N_CTX + offs_m
    # tl.store(m_ptrs, m_i)
    tl.store(H_block_ptr, q.to(matH.type.element_ty))


def qkv_fw(
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
) -> torch.Tensor:
    # batch size, number of heads, sequence length, head dimension
    BS, NH, SL, DH = matQ.shape
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = matQ.shape[-1], matK.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    HEAD_DIM_V = matV.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}

    matH = torch.empty_like(matQ)

    # grid = lambda args: (
    #     triton.cdiv(matQ.shape[2], args["BLOCK_Q"]),
    #     matQ.shape[0] * matQ.shape[1],
    #     1,
    # )
    # fix grid for debugging
    grid = lambda args: (
        triton.cdiv(matQ.shape[2], BLOCK_Q),
        matQ.shape[0] * matQ.shape[1],
        1,
    )
    print(f"Triton grid: {grid(None)}, BLOCK_Q: {BLOCK_Q}, BLOCK_KV: {BLOCK_KV}")
    vecN = torch.empty(
        (matQ.shape[0], matQ.shape[1], matQ.shape[2]),
        device=matQ.device,
        dtype=torch.float32,
    )
    vecM = torch.empty(
        (matQ.shape[0], matQ.shape[1], matQ.shape[2]),
        device=matQ.device,
        dtype=torch.float32,
    )

    _qkv_fwd[grid](
        matQ=matQ,
        matK=matK,
        matV=matV,
        qk_scale=math.sqrt(HEAD_DIM_Q),
        matH=matH,
        vecN=vecN,
        vecM=vecM,
        stride_qz=matQ.stride(0),
        stride_qh=matQ.stride(1),
        stride_qm=matQ.stride(2),
        stride_qk=matQ.stride(3),
        stride_kz=matK.stride(0),
        stride_kh=matK.stride(1),
        stride_kn=matK.stride(2),
        stride_kk=matK.stride(3),
        stride_vz=matV.stride(0),
        stride_vh=matV.stride(1),
        stride_vk=matV.stride(2),
        stride_vn=matV.stride(3),
        stride_hz=matH.stride(0),
        stride_hh=matH.stride(1),
        stride_hm=matH.stride(2),
        stride_hn=matH.stride(3),
        Z=BS,
        H=NH,
        N_CTX=SL,
        HEAD_DIM=HEAD_DIM_K,
        BLOCK_Q=BLOCK_Q,
        BLOCK_KV=BLOCK_KV,
    )

    return matH, vecM, vecN
