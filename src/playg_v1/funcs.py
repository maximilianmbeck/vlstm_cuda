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
            cls.module = load(name="playg_v2", sources=[str(filedir / "funcs.cc"), str(filedir / "funcs.cu")])
        return cls.module


cppmodule = CppModule.instance()


def func(mat_A: torch.Tensor) -> torch.Tensor:
    out = cppmodule.testkernel(mat_A)

    return out


def func2(mat_A: torch.Tensor) -> torch.Tensor:
    out = cppmodule.testkernel2(mat_A)

    return out
