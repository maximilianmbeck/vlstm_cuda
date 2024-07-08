# Results bf16 Tensor Core Kernels

**Running on my RTX4060 Laptop GPU**:

```
(xlstmpt220cu121) max@maxthinkpad:~/resourcerepos/cuda-samples/Samples/3_CUDA_Features/bf16TensorCoreGemm$ ./bf16TensorCoreGemm --kernel=0
Initializing...
GPU Device 0: "Ada" with compute capability 8.9

M: 8192 (16 x 512)
N: 8192 (16 x 512)
K: 8192 (16 x 512)
Preparing data for GPU...
Required shared memory size: 72 Kb
Computing using high performance kernel = 0 - compute_bf16gemm_async_copy
Time: 61.483009 ms
TFLOPS: 17.88
(xlstmpt220cu121) max@maxthinkpad:~/resourcerepos/cuda-samples/Samples/3_CUDA_Features/bf16TensorCoreGemm$ ./bf16TensorCoreGemm --kernel=1
Initializing...
GPU Device 0: "Ada" with compute capability 8.9

M: 8192 (16 x 512)
N: 8192 (16 x 512)
K: 8192 (16 x 512)
Preparing data for GPU...
Required shared memory size: 72 Kb
Computing using high performance kernel = 1 - compute_bf16gemm
Time: 65.975296 ms
TFLOPS: 16.67
(xlstmpt220cu121) max@maxthinkpad:~/resourcerepos/cuda-samples/Samples/3_CUDA_Features/bf16TensorCoreGemm$ ./bf16TensorCoreGemm --kernel=2
Initializing...
GPU Device 0: "Ada" with compute capability 8.9

M: 8192 (16 x 512)
N: 8192 (16 x 512)
K: 8192 (16 x 512)
Preparing data for GPU...
Required shared memory size: 72 Kb
Computing... using simple_wmma_gemm kernel
Time: 234.094589 ms
TFLOPS: 4.70
```


Differences between the Kernels: 

- simple_wmma_gemm kernel:
    - No Shared Memory, Loads directly from Global Mem

- 