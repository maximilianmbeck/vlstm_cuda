# Copyright 2023 JKU Linz
# Korbinian Pöppel, Maximilian Beck
import os

import torch
from torch.utils.cpp_extension import load as _load

# import torch.nn.functional as F
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)

print("INCLUDE:", torch.utils.cpp_extension.include_paths(cuda=True))
# print("C++ compat", torch.utils.cpp_extension.check_compiler_abi_compatibility("g++"))
# print("C compat", torch.utils.cpp_extension.check_compiler_abi_compatibility("gcc"))


# try:
#     T_MAX = int(os.environ["LSTM_T_MAX"])
# except KeyError:
T_MAX = 1 << 512

curdir = os.path.dirname(__file__)
print(curdir)

# CUDA_INCLUDE = os.environ.get("CUDA_INCLUDE", "/usr/lib")
os.environ["CUDA_LIB"] = os.path.join(
    os.path.split(torch.utils.cpp_extension.include_paths(cuda=True)[-1])[0],
    "lib",
)
print(os.environ.get("LD_LIBRARY_PATH", ""))
print(os.environ["CUDA_LIB"])


def load(*, name, sources, **kwargs):
    myargs = {
        "verbose": True,
        "with_cuda": True,
        "extra_ldflags": [f"-L{os.environ['CUDA_LIB']}", "-lcublas"],
        "extra_cuda_cflags": [
            # "-gencode",
            # "arch=compute_70,code=compute_70",
            "-gencode",
            "arch=compute_80,code=compute_80",
            "-res-usage",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
            f"-DTmax={T_MAX}",
        ],
    }
    myargs.update(**kwargs)
    return _load(name, sources, **myargs)
