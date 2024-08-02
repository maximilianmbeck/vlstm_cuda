import torch
from mlstm_bw import mlstm_bw
from torch_impl import vlstm_parallel_w_groupnorm_torch_bw as mlstm_bw_torch

S = 128  # 32 #32 #16 #8 # seq len
B = 1  # batch size
NH = 1  # num heads
DH = 128  # dim per head

DTYPE = torch.float32
DEVICE = torch.device("cuda:0")

BLOCK_Q = 16
BLOCK_KV = 16


def run_triton_mlstm_bw():

    # create qkv, inputgates, forgetgates
    torch.manual_seed(
        1
    )  # TODO from here: with seed=0 even the pytorch version alone breaks for float16 and bfloat16
    # fixed:
    # qs = torch.arange((B*NH*S*DH), device=DEVICE, dtype=DTYPE).reshape((B, NH, S, DH)) / 10.
    # qs = torch.ones((B, NH, S, DH), device=DEVICE, dtype=DTYPE) / 100.
    # ks = torch.ones((B, NH, S, DH), device=DEVICE, dtype=DTYPE) / 100.
    # # vs = torch.ones((B, NH, S, DH), device=DEVICE, dtype=DTYPE) / 100.
    # vs = torch.arange((B*NH*S*DH), device=DEVICE, dtype=DTYPE).reshape((B, NH, S, DH)) / 100.
    # # vs = torch.zeros((B, NH, S, DH), device=DEVICE, dtype=DTYPE)
    # vs[:,:,1,0] = 7.
    # qs[:,:,1,0] = 1.

    # vs[:,:,1,16] = 8.
    # qs[:,:,1,16] = 1.
    # random:
    qs = torch.randn((B, NH, S, DH), device=DEVICE, dtype=DTYPE)
    ks = torch.randn((B, NH, S, DH), device=DEVICE, dtype=DTYPE)
    vs = torch.randn((B, NH, S, DH), device=DEVICE, dtype=DTYPE)
    # igs = (1. + torch.arange((B * NH * S), device=DEVICE, dtype=DTYPE)).reshape(B, NH, S) / 10.
    # igs = torch.zeros((B, NH, S), device=DEVICE, dtype=DTYPE) #/ 10.
    igs = torch.randn((B, NH, S), device=DEVICE, dtype=DTYPE)  # / 10.
    # fgs = torch.ones((B, NH, S), device=DEVICE, dtype=DTYPE)
    # fgs = torch.ones((B, NH, S), device=DEVICE, dtype=DTYPE) *0.9 - torch.randn((B, NH, S), device=DEVICE, dtype=DTYPE) / 100.
    fgs = torch.randn((B, NH, S), device=DEVICE, dtype=DTYPE)

    dH = torch.randn((B, NH, S, DH), device=DEVICE, dtype=DTYPE)
    vecN = torch.randn((B, NH, S, 1), device=DEVICE, dtype=DTYPE)
    vecM = torch.randn((B, NH, S, 1), device=DEVICE, dtype=DTYPE)

    # inputs float16
    dtype_fp16 = torch.float16
    qs_half = qs.to(dtype=dtype_fp16)
    ks_half = ks.to(dtype=dtype_fp16)
    vs_half = vs.to(dtype=dtype_fp16)
    igs_half = igs.to(dtype=dtype_fp16)
    fgs_half = fgs.to(dtype=dtype_fp16)
    dH_half = dH.to(dtype=dtype_fp16)
    vecN_half = vecN.to(dtype=dtype_fp16)
    vecM_half = vecM.to(dtype=dtype_fp16)

    print("running torch bw")
    dQ_pt_p, dK_pt_p, dV_pt_p, dI_pt_p, dF_pt_p = mlstm_bw_torch(
        matDeltaHtilde=dH,
        matQ=qs,
        matK=ks,
        matV=vs,
        vecN=vecN,
        vecM=vecM,
        vecI=igs.unsqueeze(-1),
        vecF=fgs.unsqueeze(-1),
    )

    print("running triton bw")
    dQ_tr_p_half, dK_tr_p_half, dV_tr_p_half, dI_tr_p_half, dF_tr_p_half = mlstm_bw(
        matDeltaHtilde=dH_half,
        matQ=qs_half,
        matK=ks_half,
        matV=vs_half,
        vecN=vecN_half.squeeze(-1),
        vecM=vecM_half.squeeze(-1),
        vecI=igs_half,
        vecF=fgs_half,
        # BLOCK_Q=BLOCK_Q,
        # BLOCK_KV=BLOCK_KV,
    )


if __name__ == "__main__":
    run_triton_mlstm_bw()
