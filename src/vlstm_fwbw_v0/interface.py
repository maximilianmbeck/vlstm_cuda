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
                name="vlstm_fwbw_v0",  # name of the shared library file in the .cache/torch_extensions folder
                sources=[
                    str(filedir / "interface.cc"),
                    str(filedir / "kernel_fw.cu"),
                    str(filedir / "kernel_bw.cu"),
                ],
            )
        return cls.module


cppmodule = CppModule.instance()


from .torch_impl import vlstm_fw_torch_ref as vlstm_fw_torch


def vlstm_fw_cuda(
    mat_Q: torch.Tensor,
    mat_K: torch.Tensor,
    mat_V: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
) -> torch.Tensor:
    mat_Q = mat_Q.contiguous()
    mat_K = mat_K.contiguous()
    mat_V = mat_V.contiguous()
    igate_preact = igate_preact.contiguous()
    fgate_preact = fgate_preact.contiguous()

    out, mat_C = cppmodule.vlstm_fw(mat_Q, mat_K, mat_V, igate_preact, fgate_preact)

    return out, mat_C


def vlstm_fwbw_cuda(
    mat_Q: torch.Tensor,
    mat_K: torch.Tensor,
    mat_V: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
) -> torch.Tensor:
    mat_Q = mat_Q.contiguous()
    mat_K = mat_K.contiguous()
    mat_V = mat_V.contiguous()
    igate_preact = igate_preact.contiguous()
    fgate_preact = fgate_preact.contiguous()

    out, mat_C = vLSTMParallelFwBwCuda.apply(
        mat_Q, mat_K, mat_V, igate_preact, fgate_preact
    )

    return out, mat_C


class vLSTMParallelFwBwCuda(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        mat_Q: torch.Tensor,
        mat_K: torch.Tensor,
        mat_V: torch.Tensor,
        igate_preact: torch.Tensor,
        fgate_preact: torch.Tensor,
    ):
        mat_H, n, m, mat_C = cppmodule.vlstm_fw(
            mat_Q, mat_K, mat_V, igate_preact, fgate_preact
        )
        ctx.save_for_backward(mat_Q, mat_K, mat_V, igate_preact, fgate_preact, n, m)
        return mat_H, mat_C

    @staticmethod
    def backward(ctx, delta_H: torch.Tensor, delta_C_unused: torch.Tensor):
        mat_Q, mat_K, mat_V, igate_preact, fgate_preact, n, m = ctx.saved_tensors
        # delta_Q, delta_K, delta_V, delta_igate_preact, delta_fgate_preact = (
        #     cppmodule.vlstm_bw(
        #         delta_H, mat_Q, mat_K, mat_V, igate_preact, fgate_preact, n, m
        #     )
        # )
        # dummy init for now
        delta_Q = torch.zeros_like(mat_Q)
        delta_K = torch.zeros_like(mat_K)
        delta_V = torch.zeros_like(mat_V)
        delta_igate_preact = torch.zeros_like(igate_preact)
        delta_fgate_preact = torch.zeros_like(fgate_preact)
        return delta_Q, delta_K, delta_V, delta_igate_preact, delta_fgate_preact
