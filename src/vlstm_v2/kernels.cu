
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../util/cuda_errorcheck.h"
#include "../util/inline_ops.cuh"
#include "../util/support.h"
#include "kernel_dispatchers.h"

namespace vlstm {

namespace kernels {

/*A kernel that copies from A to B*/
template <typename scalar_t>
__global__ void copykernel(scalar_t *mat_A, scalar_t *mat_B, int r, int c);

/*MatrixMul kernel does matrix multiplication from the NVIDA cuda_samples
 * repo.*/
template <typename scalar_t, int BLOCKSIZE>
__global__ void mmkernelv1(scalar_t *matC, scalar_t *matA, scalar_t *matB,
                           int m, int n, int k);

/*MatrixMul kernel does matrix multiplication from the NVIDA cuda_samples
 * repo.*/
template <typename scalar_t, int BLOCKSIZE>
__global__ void mmkernelv2(scalar_t *matC, scalar_t *matA, scalar_t *matB,
                           int m, int n, int k);

} // namespace kernels

/* COPYKERNEL */
template <typename scalar_t>
__global__ void kernels::copykernel(scalar_t *mat_A, scalar_t *mat_B, int rdim,
                                    int cdim) {
  int cidx = blockIdx.x * blockDim.x + threadIdx.x;
  int ridx = blockIdx.y * blockDim.y + threadIdx.y;

  if (cidx < cdim && ridx < rdim) {
    int idx = ridx + cidx * rdim;
    float val = to_float<scalar_t>(mat_A[idx]);
    printf("cidx: %d, ridx: %d, val: %f\n", cidx, ridx, val);
    mat_B[idx] = mat_A[idx];
  }
}

template __global__ void
kernels::copykernel<__nv_bfloat16>(__nv_bfloat16 *mat_A, __nv_bfloat16 *mat_B,
                                   int rdim, int cdim);
template __global__ void
kernels::copykernel<__half>(__half *mat_A, __half *mat_B, int rdim, int cdim);

template <typename scalar_t>
void kernel_dispatchers::copykernel_dispatch(scalar_t *mat_A, scalar_t *mat_B,
                                             int rows, int cols) {
  printf("rows: %d, cols: %d\n", rows, cols);
  // determine the number of blocks and threads
  const dim3 block_threads(32, 32);
  const dim3 grid_blocks((cols + block_threads.x - 1) / block_threads.x,
                         (rows + block_threads.y - 1) / block_threads.y);
  printf("blocksxy: %d-%d, threads: %d-%d\n", grid_blocks.x, grid_blocks.y,
         block_threads.x, block_threads.y);
  kernels::copykernel<scalar_t>
      <<<grid_blocks, block_threads>>>(mat_A, mat_B, rows, cols);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
}

template void kernel_dispatchers::copykernel_dispatch<__nv_bfloat16>(
    __nv_bfloat16 *mat_A, __nv_bfloat16 *mat_B, int rows, int cols);
template void kernel_dispatchers::copykernel_dispatch<__half>(__half *mat_A,
                                                              __half *mat_B,
                                                              int rows,
                                                              int cols);

////////////////////////////////////////////////////////////////////////////////////////
/* MATRIXMUL KERNEL V1*/
/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * k is A's width and n is B's width
 * works only if m, n, k are divisible by BLOCK_SIZE
 */
template <typename scalar_t, int BLOCK_SIZE>
__global__ void kernels::mmkernelv1(scalar_t *C, scalar_t *A, scalar_t *B,
                                    int m, int n, int k) {
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  // int tIdx = threadIdx.x + blockDim.x * threadIdx.y;

  int cx = bx * blockDim.x + tx;
  int cy = by * blockDim.y + ty;
  if ((cx == 0) && (cy == 0)) {
    printf("In Kernel: m: %d, n: %d, k: %d\n", m, n, k);
    printf("In Kernel: gdim.x: %d, gdim.y: %d, bdim.x: %d, bdim.y: %d\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);
  }
  if (false) {
    // printf("bdx: %d, bdy: %d, b_cx: %d, b_cy: %d, tIdx: %d\n", blockDim.x,
    //        blockDim.y, bx * blockDim.x + tx, by * blockDim.y + ty, tIdx);
    printf("cx: %d, cy: %d\n", cx, cy);
  }
  // @max Output index of the matrix C
  int block_cIdx = n * BLOCK_SIZE * by + BLOCK_SIZE * bx; // (blocklevel)
  int thread_cIdx = block_cIdx + n * ty + tx;             // (threadlevel)

  // Index of the first sub-matrix of A processed by the block
  int aBegin = k * BLOCK_SIZE * by; // (blocklevel)

  // Index of the last sub-matrix of A processed by the block
  int aEnd = aBegin + k - 1; // (blocklevel)

  // Step size used to iterate through the sub-matrices of A
  int aStep = BLOCK_SIZE; // (blocklevel)

  // Index of the first sub-matrix of B processed by the block
  int bBegin = BLOCK_SIZE * bx; // (blocklevel)

  // Step size used to iterate through the sub-matrices of B
  int bStep = BLOCK_SIZE * n; // (blocklevel)

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  // @max: Csub is also used to accumulate the result for one entry in the
  // output matrix C
  scalar_t Csub = dscalar_zero<scalar_t>();

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  // @max: outer loop, progresses always in BLOCK_SIZE steps and accumulates
  // the final values in Csub
  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ scalar_t As[BLOCK_SIZE][BLOCK_SIZE];
    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ scalar_t Bs[BLOCK_SIZE][BLOCK_SIZE];
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // @max: There are three bounds to consider:
    // TODO rewrite kernel with true indices not with memory offsets
    // - m & n in the B matrix: the tread block bounds
    // - k in the A and B matrix: the bounds of the loop over the k dimension
    // TODO @max: WRONG! the bounds of the loop over the k dimension are not
    // correct! rewrite!
    As[ty][tx] = A[a + k * ty + tx]; // (threadlevel)
    Bs[ty][tx] = B[b + n * ty + tx]; // (threadlevel)
    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll
    // @max: inner looop
    for (int i = 0; i < BLOCK_SIZE; ++i) {
      Csub = add_g(Csub, mul_g(As[ty][i],
                               Bs[i][tx])); // (threadlevel): each thread
                                            // operates on BLOCK_SIZE elements
    }
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }
  // Write the block sub-matrix to device memory;
  // each thread writes one element
  C[thread_cIdx] = Csub; // (threadlevel)
}

/*
A: (m x k)
B: (k x n)
C: (m x n)
*/
template <typename scalar_t>
void kernel_dispatchers::mmkernelv1_dispatch(scalar_t *matC, scalar_t *matA,
                                             scalar_t *matB, int m, int n,
                                             int k) {
  printf("m: %d, n: %d, k: %d\n", m, n, k);
  const int BLOCK_SIZE = 8;

  // determine the number of blocks and threads
  const dim3 blockDims(BLOCK_SIZE, BLOCK_SIZE);
  const dim3 gridDims((n + blockDims.x - 1) / blockDims.x,
                      (m + blockDims.y - 1) / blockDims.y);
  printf("blocksxy: %d-%d, threads: %d-%d\n", gridDims.x, gridDims.y,
         blockDims.x, blockDims.y);

  // if (m % BLOCK_SIZE != 0 || n % BLOCK_SIZE != 0 || k % BLOCK_SIZE != 0) {
  //     printf("m, n, k must be divisible by BLOCK_SIZE\n");
  //     return;
  // }
  kernels::mmkernelv1<scalar_t, BLOCK_SIZE>
      <<<gridDims, blockDims>>>(matC, matA, matB, m, n, k);

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
}

template void kernel_dispatchers::mmkernelv1_dispatch<__nv_bfloat16>(
    __nv_bfloat16 *matC, __nv_bfloat16 *matA, __nv_bfloat16 *matB, int m, int n,
    int k);
template void kernel_dispatchers::mmkernelv1_dispatch<__half>(
    __half *matC, __half *matA, __half *matB, int m, int n, int k);

} // namespace vlstm