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
                name="vlstm_fw_v0",
                sources=[str(filedir / "interface.cc"), str(filedir / "kernels.cu")],
            )
        return cls.module


cppmodule = CppModule.instance()


from .torch_impl import vlstm_fw_torch_ref as vlstm_fw_torch


def vlstm_fw_cuda(
    mat_Q: torch.Tensor, mat_K: torch.Tensor, mat_V: torch.Tensor
) -> torch.Tensor:
    mat_Q = mat_Q.contiguous()
    mat_K = mat_K.contiguous()
    mat_V = mat_V.contiguous()

    out = cppmodule.vlstm_fw(mat_Q, mat_K, mat_V)

    return out
