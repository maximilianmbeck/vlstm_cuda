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
                name="mm_v0",
                sources=[str(filedir / "interface.cc"), str(filedir / "kernels.cu")],
            )
        return cls.module


cppmodule = CppModule.instance()


def testkernel(mat_A: torch.Tensor) -> torch.Tensor:
    out = cppmodule.testkernel(mat_A)

    return out


def copykernel(mat_A: torch.Tensor) -> torch.Tensor:
    out = cppmodule.copykernel(mat_A)

    return out


def mmkernelv1(mat_A: torch.Tensor, mat_B: torch.Tensor) -> torch.Tensor:
    out = cppmodule.mmkernelv1(mat_A, mat_B)

    return out


def mmkernelv2(mat_A: torch.Tensor, mat_B: torch.Tensor) -> torch.Tensor:
    out = cppmodule.mmkernelv2(mat_A, mat_B)

    return out
