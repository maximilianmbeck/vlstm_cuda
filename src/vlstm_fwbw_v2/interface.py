# Copyright JKU Linz 2023
# Author: Maximilian Beck

import os
from pathlib import Path

import torch

from ..cuda_init import load

filedir = Path(os.path.dirname(os.path.abspath(__file__)))


class CppModule(object):
    module = None

    @classmethod
    def instance(cls):
        if cls.module is None:
            cls.module = load(
                name="vlstm_fwbw_v2",  # name of the shared library file in the .cache/torch_extensions folder
                sources=[
                    str(filedir / "interface.cc"),
                    str(filedir / "kernel_fw.cu"),
                    str(filedir / "kernel_bw.cu"),
                ],
            )
        return cls.module


cppmodule = CppModule.instance()


from .torch_impl import vlstm_fw_torch_ref as vlstm_fw_torch
from .torch_impl import vlstm_parallel_bw_torch_w_groupnorm as vlstm_bw_torch_obw
from .torch_impl import (
    vlstm_parallel_fw_torch_w_groupnorm as vlstm_fwbw_torch_autogradbw,
)
from .torch_impl import vlstm_parallel_fwbw_torch_w_groupnorm as vlstm_fwbw_torch_obw


def vlstm_fw_cuda(
    mat_Q: torch.Tensor,
    mat_K: torch.Tensor,
    mat_V: torch.Tensor,
    vec_igp: torch.Tensor,
    vec_fgp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, NH, S, DH = mat_Q.size()
    mat_Q = mat_Q.contiguous()
    mat_K = mat_K.contiguous()
    mat_V = mat_V.contiguous()
    vec_igp = vec_igp.contiguous()
    vec_fgp = vec_fgp.contiguous()

    # allocate outputs and set to zero
    mat_H = torch.ones_like(mat_Q)
    vec_n = torch.zeros((B, NH, S, 1), dtype=mat_Q.dtype, device=mat_Q.device)
    vec_m = torch.zeros((B, NH, S, 1), dtype=mat_Q.dtype, device=mat_Q.device)

    # only for debugging
    mat_C = torch.ones((B, NH, S, S), dtype=mat_Q.dtype, device=mat_Q.device)

    cppmodule.vlstm_fw(
        mat_H, vec_n, vec_m, mat_C, mat_Q, mat_K, mat_V, vec_igp, vec_fgp
    )

    return mat_H, vec_n, vec_m, mat_C


def vlstm_bw_cuda(
    mat_delta_H: torch.Tensor,
    mat_Q: torch.Tensor,
    mat_K: torch.Tensor,
    mat_V: torch.Tensor,
    vec_igp: torch.Tensor,
    vec_fgp: torch.Tensor,
    vec_n: int,
    vec_m: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    B, NH, S, DH = mat_Q.size()
    mat_delta_H = mat_delta_H.contiguous()
    mat_Q = mat_Q.contiguous()
    mat_K = mat_K.contiguous()
    mat_V = mat_V.contiguous()
    vec_igp = vec_igp.contiguous()
    vec_fgp = vec_fgp.contiguous()

    # allocate outputs and set to zero
    mat_delta_Q = torch.zeros_like(mat_Q)
    mat_delta_K = torch.zeros_like(mat_K)
    mat_delta_V = torch.zeros_like(mat_V)
    vec_delta_igp = torch.zeros_like(vec_igp)
    vec_delta_fgp = torch.zeros_like(vec_fgp)

    # only for debugging
    mat_C = torch.zeros((B, NH, S, S), dtype=mat_Q.dtype, device=mat_Q.device)

    cppmodule.vlstm_bw(
        mat_delta_Q,
        mat_delta_K,
        mat_delta_V,
        vec_delta_igp,
        vec_delta_fgp,
        mat_C,
        mat_delta_H,
        mat_Q,
        mat_K,
        mat_V,
        vec_igp,
        vec_fgp,
        vec_n,
        vec_m,
    )
    return (mat_delta_Q, mat_delta_K, mat_delta_V, vec_delta_igp, vec_delta_fgp, mat_C)


def vlstm_fwbw_cuda(
    mat_Q: torch.Tensor,
    mat_K: torch.Tensor,
    mat_V: torch.Tensor,
    vec_igp: torch.Tensor,
    vec_fgp: torch.Tensor,
) -> torch.Tensor:
    mat_Q = mat_Q.contiguous()
    mat_K = mat_K.contiguous()
    mat_V = mat_V.contiguous()
    vec_igp = vec_igp.contiguous()
    vec_fgp = vec_fgp.contiguous()

    mat_H, vec_n, vec_m, mat_C = vLSTMParallelFwBwCuda.apply(
        mat_Q, mat_K, mat_V, vec_igp, vec_fgp
    )

    return mat_H, vec_n, vec_m, mat_C


class vLSTMParallelFwBwCuda(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        mat_Q: torch.Tensor,
        mat_K: torch.Tensor,
        mat_V: torch.Tensor,
        vec_igp: torch.Tensor,
        vec_fgp: torch.Tensor,
    ):
        mat_H, vec_n, vec_m, mat_C = vlstm_fw_cuda(
            mat_Q, mat_K, mat_V, vec_igp, vec_fgp
        )
        ctx.save_for_backward(mat_Q, mat_K, mat_V, vec_igp, vec_fgp, vec_n, vec_m)
        return mat_H, vec_n, vec_m, mat_C

    @staticmethod
    def backward(
        ctx,
        delta_H: torch.Tensor,
        delta_n_unused: torch.Tensor,
        delta_m_unused: torch.Tensor,
        delta_C_unused: torch.Tensor,
    ):
        mat_Q, mat_K, mat_V, vec_igp, vec_fgp, vec_n, vec_m = ctx.saved_tensors
        (delta_Q, delta_K, delta_V, delta_igate_preact, delta_fgate_preact, mat_C) = (
            vlstm_bw_cuda(delta_H, mat_Q, mat_K, mat_V, vec_igp, vec_fgp, vec_n, vec_m)
        )
        return delta_Q, delta_K, delta_V, delta_igate_preact, delta_fgate_preact
