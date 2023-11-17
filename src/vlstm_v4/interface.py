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
                name="vlstm_v4",
                sources=[str(filedir / "interface.cc"), str(filedir / "kernels.cu")],
            )
        return cls.module


cppmodule = CppModule.instance()


def mmkernelv1(mat_A: torch.Tensor, mat_B: torch.Tensor) -> torch.Tensor:
    out = cppmodule.mmkernelv1(mat_A, mat_B)

    return out


def qkvkernel(
    mat_Q: torch.Tensor, mat_K: torch.Tensor, mat_V: torch.Tensor
) -> torch.Tensor:
    mat_Q = mat_Q.contiguous()
    # transpose K matrix in order to have coalesced access in the kernel
    # Could do this also on C++ side
    mat_K = mat_K.contiguous()
    mat_V = mat_V.contiguous()

    out = cppmodule.qkvkernel(mat_Q, mat_K, mat_V)

    return out
