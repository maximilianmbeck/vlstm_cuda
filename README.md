# vLSTM_cuda

In this repo we implement the vLSTM forward and backward pass in CUDA.
It aims to show the progress of the different kernels.

## Kernel versions in chronological order

1. **Kernel structure setup:**

- playg_v1:
- playg_v2: Implement kernel calls from C++ with different dtypes. Setup simple structure. Integrate to PyTorch via pybind11.

2. **Getting to know simple matrix multiplication in CUDA:**

- matrixMul: matrixMul example from the cuda_samples repo.
- mm_v0: Bring cuda_sample in our folder structure.
- mm_v1: Play around with matrix multiplication.

3. **Implement a $QK^\top V$ (double) matrix multiplication without tensor cores**:

- qkv_v0: First attempt. Wrong computation partitioning. The whole grid was used for single Q and KV tiles. Did not work as we had different parts of the tile in different shared memories of the grid-blocks (streaming multiprocessors).
- qkv_v1: Second attempt. Now corrected work partitioning. Use one grid block / streaming multiprocessor per Q tile.

4. **Implement vLSTM forward pass without tensor cores**:

- vlstm_fw_v0: Build on qkv_v1, make qkv_v1 causal, i.e. compute
$$
(QK^\top \odot D) \ V \ ,
$$
where $D$ is a lower triangular matrix (ones) and the upper triangle are zeros.

- vlstm_fw_v1: Build on vlstm_fw_v0, integrate forget&input gates + normalization. This is the first fused vlstm kernel for the forward pass only.
  - Open **TODO**: Optimize the fgate cumsum computation over the grid iterations. Do not recompute from scratch, but reuse previous computation.

5. **Implement vLSTM backward pass without tensor cores**:

- vlstm_fwbw_v0: Build on vlstm_fw_v1, implement the backward pass using the same tiling strategy as FlashAttention2.
  - adapt forward to store the max state and the n state in HBM for reuse in backward.
  - make forward more efficient during gate matrix computation (reuse previous computations/sums of forgetgates):
    - we sync the thread blocks in qTileDim direction over HBM. 

- vlstm_fwbw_v1: Build on vlstm_fwbw_v0, Increase / Tune Block & Grid dims. 
   - **TODOs**:
   - Write combined numerical correctness checks. OK
   - Check correctness for other dtypes. OK
   - Allocate temp global memory in kernel. OK
   - Check larger tile sizes + grid/threadblock dims TODO
   - Add speed tests. TODO
   - **Worklog**:
   - fix bug on multiple iterations backward kernel does not match for delta Q & delta K anymore. OK
      - Possible fixes:
        1. race condition: add synchronization around delta K + delta Q calculation (probably gridGroup.sync())
          - mat P and R is always correct
          - -> bug some where around deltaHtile or qTile loading 
          - -> bug some where around writing the result to HBM
          - !!! adding syncs did not help so far. Error even persists if we have only 1 thread block, suspect something with memory allocation, move on to 2.
        2. allocate memory in python and pass it to the kernel (no HBM allocations on C side (besides the temporary ones within the kernel with cudaMalloc))
          - But if it is memory allocation why then only for Q, K ???
          - !!! allocating the memory on python side also did not help
        3. bug somewhere in kernel?
          - yes, it is a bug in the kernel. -> find it!!
          - -> BUG FIXED! The deltaK Tile and deltaV Tile were not initialized in shared memory

    - fix half precision bug (numeric values did not match):
      - The kernel algorithm was correct. 
      - in inline_ops_fp16.cuh there was the definition of log_g(__half x) missing.

    - bug: output hs in forward pass has nans for large sizes, only the last few entries (rows)
      - The nan stem from very large normalizer values for low kvTileIdxes, leading to infs (overflow)
      - Reason: the first columns in the d matrix (lower left corner) contain large negative values, 
        as first m_val we choose the max of this tile which is still very large. Then 
        $n=max(|l|, e^{-m}) $ where $e^{-m}$ gets inf as $m<<0$.
      - Solution: bound the m val from below by a specific value (currently -10, exp(10)=22026, within fp16 range)
    
    - bug: larger QtileDim and KVtileDim do not work yet

    - possible bug / instability: still nan in fp16 with S=64, seed=0
      - Reason: the weighting factor for h_prev was computed suboptimally
      - Solution: compute it in the following order $(e^{m1-m2}*n1)/n2$ instead of computing the fraction first

    
6. **Implement vLSTM forward pass with tensor cores**:

7. **Implement vLSTM backward pass with tensor cores**:



## CUDA questions

**Synchronization**:
- How can I synchronize only parts of the gridGroup? Or how can I define gridGroups as subset of the whole grid. 
  - Example: I want to synchronize only the blocks that are used for one batch head. Otheres should be independent.
- Where to put __syncthreads(); ?


## CUDA Resources

### CUDA Samples 
> https://github.com/NVIDIA/cuda-samples/tree/master

Notes: 

- When compiling CUDA samples ensure correct C++ standard and linking:
```
make NVCCFLAGS="-std=c++11" LDFLAGS="-lstdc++" VERBOSE=1
```
- Also possible:
``` 
# Set the HOST_COMPILER explicitly
make HOST_COMPILER=/home/max/miniconda3/envs/xlstmpt220cu121/bin/x86_64-conda-linux-gnu-gcc
```
- See also: https://chatgpt.com/c/03784eca-c557-4896-b561-984d91bbaf31 

## Some Facts about GPUs

### RTX 4060 Laptop GPU
```
 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA GeForce RTX 4060 Laptop GPU"
  CUDA Driver Version / Runtime Version          12.3 / 12.2
  CUDA Capability Major/Minor version number:    8.9
  Total amount of global memory:                 7940 MBytes (8325824512 bytes)
  (024) Multiprocessors, (128) CUDA Cores/MP:    3072 CUDA Cores
  GPU Max Clock rate:                            2250 MHz (2.25 GHz)
  Memory Clock rate:                             8001 Mhz
  Memory Bus Width:                              128-bit
  L2 Cache Size:                                 33554432 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        102400 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 12.3, CUDA Runtime Version = 12.2, NumDevs = 1
```