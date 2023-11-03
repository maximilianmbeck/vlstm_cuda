#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../util/support.h"
#include "../util/inline_ops.cuh"
#include "../util/cuda_errorcheck.h"
#include "kernel_dispatchers.h"

namespace vlstm {

namespace kernels {

/*A kernel that copies from A to B*/
template <typename scalar_t>
__global__ void copykernel(scalar_t *mat_A, scalar_t *mat_B, int r,
                           int c);

/*MatrixMul kernel does matrix multiplication from the NVIDA cuda_samples
 * repo.*/
template<int> __global__ void mmkernelv1(__nv_bfloat16 *matC, __nv_bfloat16 *matA,
                           __nv_bfloat16 *matB, int wA, int wB);

} // namespace kernels

/* COPYKERNEL */
template <typename scalar_t>
__global__ void kernels::copykernel<scalar_t>(scalar_t *mat_A, scalar_t *mat_B,
                                    int rdim, int cdim) {
  int cidx = blockIdx.x * blockDim.x + threadIdx.x;
  int ridx = blockIdx.y * blockDim.y + threadIdx.y;

  if (cidx < cdim && ridx < rdim) {
    int idx = ridx + cidx * rdim;
    float val = to_float<scalar_t>(mat_A[idx]);
    printf("cidx: %d, ridx: %d, val: %f\n", cidx, ridx, val);
    mat_B[idx] = mat_A[idx];
  }
}

template __global__ void kernels::copykernel<__nv_bfloat16>(__nv_bfloat16 *mat_A, __nv_bfloat16 *mat_B, int rdim, int cdim);
template __global__ void kernels::copykernel<__half>(__half *mat_A, __half *mat_B, int rdim, int cdim);

template <typename scalar_t> // todo remove this scalar_t after func name
void kernels::copykernel_dispatch<scalar_t>(scalar_t *mat_A, scalar_t *mat_B,
                                  int rows, int cols) {
  printf("rows: %d, cols: %d\n", rows, cols);
  // determine the number of blocks and threads
  const dim3 block_threads(32, 32);
  const dim3 grid_blocks((rows + block_threads.y - 1) / block_threads.y,
                         (cols + block_threads.x - 1) / block_threads.x);
  printf("blocksxy: %d-%d, threads: %d-%d\n", grid_blocks.x, grid_blocks.y,
         block_threads.x, block_threads.y);
  kernels::copykernel<scalar_t><<<grid_blocks, block_threads>>>(mat_A, mat_B, rows, cols);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
}

template void kernels::copykernel_dispatch<__nv_bfloat16>(__nv_bfloat16 *mat_A, __nv_bfloat16 *mat_B, int rows, int cols);
template void kernels::copykernel_dispatch<__half>(__half *mat_A, __half *mat_B, int rows, int cols);

/* MATRIXMUL KERNEL */
/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <typename scalar_t, int BLOCK_SIZE>
__global__ void kernels::mmkernelv1(scalar_t *C, scalar_t *A,
                                    scalar_t *B, int wA, int wB) {
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Index of the first sub-matrix of A processed by the block
  int aBegin = wA * BLOCK_SIZE * by;

  // Index of the last sub-matrix of A processed by the block
  int aEnd = aBegin + wA - 1;

  // Step size used to iterate through the sub-matrices of A
  int aStep = BLOCK_SIZE;

  // Index of the first sub-matrix of B processed by the block
  int bBegin = BLOCK_SIZE * bx;

  // Step size used to iterate through the sub-matrices of B
  int bStep = BLOCK_SIZE * wB;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  scalar_t Csub = dscalar_zero<scalar_t>();

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
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
    As[ty][tx] = A[a + wA * ty + tx];
    Bs[ty][tx] = B[b + wB * ty + tx];

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll

    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub = add_g(Csub, mul_g(As[ty][k], Bs[k][tx]));
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  C[c + wB * ty + tx] = Csub;
}

/*
A: (m x k)
B: (k x n)
C: (m x n)
*/
template <typename scalar_t>
void kernels::mmkernelv1_dispatch(scalar_t *matC, scalar_t *matA,
                                  scalar_t *matB, int m, int n, int k) {
  const int BLOCK_SIZE = 32;

  printf("m: %d, n: %d, k: %d\n", m, n, k);
  if (m % BLOCK_SIZE != 0 || n % BLOCK_SIZE != 0 || k % BLOCK_SIZE != 0) {
      printf("m, n, k must be divisible by BLOCK_SIZE\n");
      return;
    }

  // determine the number of blocks and threads
  const dim3 block_threads(BLOCK_SIZE, BLOCK_SIZE);
  const dim3 grid_blocks((m + block_threads.y - 1) / block_threads.y,
                         (n + block_threads.x - 1) / block_threads.x);
  printf("blocksxy: %d-%d, threads: %d-%d\n", grid_blocks.x, grid_blocks.y,
         block_threads.x, block_threads.y);
  kernels::mmkernelv1<scalar_t, BLOCK_SIZE>
      <<<grid_blocks, block_threads>>>(matC, matA, matB, k, n);

  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

}

template void kernels::mmkernelv1_dispatch<__nv_bfloat16>(__nv_bfloat16 *matC, __nv_bfloat16 *matA, __nv_bfloat16 *matB, int m, int n, int k);
template void kernels::mmkernelv1_dispatch<__half>(__half *matC, __half *matA, __half *matB, int m, int n, int k);

} // namespace vlstm