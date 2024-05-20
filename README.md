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

- **TODO** vlstm_fw_v1: Build on vlstm_fw_0, integrate forget&input gates + normalization. This should be a first fused vlstm kernel. 
