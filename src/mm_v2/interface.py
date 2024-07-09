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
                name="mm_v2",
                sources=[str(filedir / "interface.cc"), str(filedir / "kernels.cu")],
            )
        return cls.module


cppmodule = CppModule.instance()


def mmkernel(
    mat_A: torch.Tensor, mat_B: torch.Tensor, mat_C: torch.Tensor
) -> torch.Tensor:
    mat_C = mat_C.to(dtype=torch.float32)
    out = cppmodule.mmkernel(mat_A, mat_B, mat_C)

    return out
