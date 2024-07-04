import torch
from tqdm import tqdm

from src.vlstm_fwbw_v1.interface import (
    vlstm_bw_cuda,
    vlstm_bw_torch_obw,
    vlstm_fw_cuda,
    vlstm_fw_torch,
    vlstm_fwbw_cuda,
    vlstm_fwbw_torch_obw,
)


# Run forward kernel multiple times and check if the results are consistent
def check_multi_kernel_calls_fw(q, k, v, ig, fg, num_calls, atol, rtol):
    # pytorch baseline
    hs_pt, n_pt, m_pt, _, matD_pt = vlstm_fw_torch(
        queries=q, keys=k, values=v, igate_preact=ig, fgate_preact=fg
    )

    hs_correct = []
    n_correct = []
    m_correct = []
    matD_correct = []
    for i in tqdm(range(num_calls)):
        # run kernel
        hs_cu, n_cu, m_cu, matD_cu = vlstm_fw_cuda(
            mat_Q=q, mat_K=k, mat_V=v, vec_igp=ig, vec_fgp=fg
        )

        hs_correct.append(torch.allclose(hs_cu, hs_pt, rtol=rtol, atol=atol))
        n_correct.append(torch.allclose(n_cu, n_pt, rtol=rtol, atol=atol))
        m_correct.append(torch.allclose(m_cu, m_pt, rtol=rtol, atol=atol))
        matD_correct.append(
            torch.allclose(
                (matD_cu - matD_pt).tril(),
                torch.zeros_like((matD_cu)),
                rtol=rtol,
                atol=atol,
            )
        )

    print(f"hs correct/total: {sum(hs_correct)}/{num_calls}")
    print(f"n correct/total: {sum(n_correct)}/{num_calls}")
    print(f"m correct/total: {sum(m_correct)}/{num_calls}")
    print(f"matD correct/total: {sum(matD_correct)}/{num_calls}")


# Run forward / backward kernel multiple times and check if the results are consistent
def check_multi_kernel_calls_fwbw(dH, q, k, v, ig, fg, num_calls, atol, rtol):
    # pytorch baseline
    hs_pt, n_pt, m_pt, _, matD_pt = vlstm_fw_torch(
        queries=q, keys=k, values=v, igate_preact=ig, fgate_preact=fg
    )
    (
        dQs_pt,
        dKs_pt,
        dVs_pt,
        dIgs_pt,
        dFgs_pt,
        delta_D_pt,
        delta_Dtilde_pt,
        delta_fbar_pt,
        mat_P_pt,
        mat_R_pt,
    ) = vlstm_bw_torch_obw(
        delta_Htilde=dH,
        queries=q,
        keys=k,
        values=v,
        igate_preact=ig,
        fgate_preact=fg,
        var_n=n_pt,
        var_m=m_pt,
    )

    hs_correct = []
    n_correct = []
    m_correct = []
    matD_correct = []

    dQ_correct = []
    dK_correct = []
    dV_correct = []
    dIg_correct = []
    dFg_correct = []

    for i in tqdm(range(num_calls)):
        # run kernel
        hs_cu, n_cu, m_cu, matD_cu = vlstm_fw_cuda(
            mat_Q=q, mat_K=k, mat_V=v, vec_igp=ig, vec_fgp=fg
        )
        (
            dQs_cu,
            dKs_cu,
            dVs_cu,
            dIgs_cu,
            dFgs_cu,
            matC_cu,
        ) = vlstm_bw_cuda(
            mat_delta_H=dH,
            mat_Q=q,
            mat_K=k,
            mat_V=v,
            vec_igp=ig,
            vec_fgp=fg,
            vec_n=n_cu,
            vec_m=m_cu,
        )

        hs_correct.append(torch.allclose(hs_cu, hs_pt, rtol=rtol, atol=atol))
        n_correct.append(torch.allclose(n_cu, n_pt, rtol=rtol, atol=atol))
        m_correct.append(torch.allclose(m_cu, m_pt, rtol=rtol, atol=atol))
        matD_correct.append(
            torch.allclose(
                (matD_cu - matD_pt).tril(),
                torch.zeros_like((matD_cu)),
                rtol=rtol,
                atol=atol,
            )
        )

        dQ_correct.append(torch.allclose(dQs_cu, dQs_pt, rtol=rtol, atol=atol))
        dK_correct.append(torch.allclose(dKs_cu, dKs_pt, rtol=rtol, atol=atol))
        dV_correct.append(torch.allclose(dVs_cu, dVs_pt, rtol=rtol, atol=atol))
        dIg_correct.append(torch.allclose(dIgs_cu, dIgs_pt, rtol=rtol, atol=atol))
        dFg_correct.append(torch.allclose(dFgs_cu, dFgs_pt, rtol=rtol, atol=atol))

    print("===== FW =====")
    print(f"hs correct/total: {sum(hs_correct)}/{num_calls}")
    print(f"n correct/total: {sum(n_correct)}/{num_calls}")
    print(f"m correct/total: {sum(m_correct)}/{num_calls}")
    print(f"matD correct/total: {sum(matD_correct)}/{num_calls}")
    print("===== BW =====")
    print(f"dQ correct/total: {sum(dQ_correct)}/{num_calls}")
    print(f"dK correct/total: {sum(dK_correct)}/{num_calls}")
    print(f"dV correct/total: {sum(dV_correct)}/{num_calls}")
    print(f"dIg correct/total: {sum(dIg_correct)}/{num_calls}")
    print(f"dFg correct/total: {sum(dFg_correct)}/{num_calls}")
