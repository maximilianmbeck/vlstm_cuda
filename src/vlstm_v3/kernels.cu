#include <cassert>

#include <cstdio>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <driver_types.h>

#include "../util/cuda_errorcheck.h"
#include "../util/inline_ops.cuh"
#include "../util/inline_print.cuh"
#include "../util/support.h"
#include "kernel_dispatchers.h"

#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

namespace vlstm {

namespace kernels {

/* MatrixMul kernel does matrix multiplication from the NVIDA cuda_samples
 repo.*/
template <typename scalar_t, int BLOCKSIZE>
__global__ void mmkernelv1(scalar_t *matC, scalar_t *matA, scalar_t *matB,
                           int m, int n, int k);

template <typename scalar_t, int TblockDim, int QblockDim, int KVblockDim>
__global__ void qkvkernel(scalar_t *matC, scalar_t *matQ, scalar_t *matK,
                          scalar_t *matV, int batchSize, int numHeads,
                          int seqLen, int dimHeads);

} // namespace kernels

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
      if ((cx == 0) && (cy == 9)) {
        print_val("(ty,i) - As", ty, i, As[ty][i]);
        print_val("(ty,i) - Bs", ty, i, Bs[ty][i]);
      }
      Csub = add_g(Csub, mul_g(As[ty][i],
                               Bs[i][tx])); // (threadlevel): each thread
                                            // operates on BLOCK_SIZE elements
      if ((cx == 0) && (cy == 9)) {
        print_val("(cx,cy)-InLoop:Csub", cx, cy, Csub);
      }
    }
    if ((cx == 0) && (cy == 9)) {
      print_val("(cx,cy)-AfterLoop:Csub", cx, cy, Csub);
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
  const int BLOCK_SIZE = 4;

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

////////////////////////////////////////////////////////////////////////////////////////

#define TBLOCK_DIM 4 // TblockDim: corresponds to BLOCK_DIM in matmul
#define QTILE_DIM 8  // QtileDim: TileDim for Q along seqLen dim
#define KVTILE_DIM 8 // KVtileDim: TileDim for K&V along seqLen dim

// TODO use dynamic shared memory!
#define HD_SIZE                                                                \
  64 // HD_SIZE: size of the allocated shared memory for the hidden dim

#define DEBUG 1
#define DEBUG2 1

/* QKV Kernel v1 */

template <typename scalar_t, int TblockDim, int QtileDim, int KVtileDim>
__global__ void kernels::qkvkernel(scalar_t *matC, scalar_t *matQ,
                                   scalar_t *matK, scalar_t *matV,
                                   int batchSize, int numHeads, int seqLen,
                                   int dimHeads) {
  // int tIdx = threadIdx.x + blockDim.x * threadIdx.y;
#ifdef DEBUG
  if ((blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0) &&
      (threadIdx.y == 0)) {
    printf("In Kernel: gdim.x: %d, gdim.y: %d, gdim.z: %d, bdim.x: %d, bdim.y: "
           "%d\n",
           gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y);
  }
#endif

  // Most outer loop: Loop over batchSize * numHeads (can be parallelized later
  // along gridDim.z)
  const uint batchHeadStep = seqLen * dimHeads;
  const uint batchHeadEnd = batchSize * numHeads * batchHeadStep;
  //! Note: batchHeadIdx indices into global memory
  for (uint batchHeadIdx = 0; batchHeadIdx < batchHeadEnd;
       batchHeadIdx += batchHeadStep) {

    // access to Q & V (not transposed (S, DH))
    const uint qvBlockIdx = batchHeadIdx + dimHeads * TblockDim * blockIdx.y +
                            TblockDim * blockIdx.x;
    const uint qvThreadIdx = qvBlockIdx + dimHeads * threadIdx.y + threadIdx.x;

    // access to K (K is transposed (DH, S))
    const uint kBlockIdx =
        batchHeadIdx + seqLen * TblockDim * blockIdx.x + TblockDim * blockIdx.y;
    const uint kThreadIdx = kBlockIdx + seqLen * threadIdx.y + threadIdx.x;

#ifdef DEBUG2
    if ((threadIdx.x == 0) && (threadIdx.y == 2)) {
      print_val("(bx,by) - Q", blockIdx.x, blockIdx.y, matQ[qvThreadIdx]);
      // print_val("(ty,i) - Bs", ty, i, Bs[ty][i]);
    }
    matC[qvThreadIdx] = matQ[qvThreadIdx];
    __syncthreads();
#endif

    // Ends for looplevel 1&2:
    uint qTileEnd = seqLen / QtileDim;
    uint kvTileEnd = seqLen / KVtileDim;
    // looplevel 1: loop over Qtile blocks along seqLen dim
    // Note: qTileIdx does not index into global memory
    for (uint qTileIdx = 0; qTileIdx < qTileEnd; qTileIdx++) {

      // init qTile in shared memory
      // TODO use dynamic shared memory!
      __shared__ scalar_t qTile[QtileDim][HD_SIZE];

      // compute indices
      // left upper corner of qTileBlock in Q
      const uint qTileBlockIdx = batchHeadIdx +
                                 qTileIdx * dimHeads * TblockDim * blockIdx.y +
                                 TblockDim * blockIdx.x;
      // TODO: thread index

      // load qTile into shared memory

      // initialize result cTile in shared memory

      __syncthreads();

      // looplevel 2: loop over KVtile blocks along seqLen dim
      for (uint kvTileIdx = 0; kvTileIdx < kvTileEnd; kvTileIdx++) {

        // load kTile into shared memory
        // load vTile into shared memory

        // initialize sTile in shared memory for intermediate result of QK^T

        // compute S = QK^T, i.e. fill sTile

        // compute C += S * V, i.e. fill cTile

        __syncthreads();
      }

      // write cTile to global memory
    }

    // const uint cx = bx * blockDim.x + tx;
    // const uint cy = by * blockDim.y + ty;

    // loop over batchSize * numHeads
    // if ((cx == 0) && (cy == 0)) {
    //   printf("In Kernel: m: %d, n: %d, k: %d\n", m, n, k);
    //   printf("In Kernel: gdim.x: %d, gdim.y: %d, bdim.x: %d, bdim.y: %d\n",
    //          gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    // }
    // if (false) {
    //   // printf("bdx: %d, bdy: %d, b_cx: %d, b_cy: %d, tIdx: %d\n",
    //   blockDim.x,
    //   //        blockDim.y, bx * blockDim.x + tx, by * blockDim.y + ty,
    //   tIdx); printf("cx: %d, cy: %d\n", cx, cy);
    // }
    // int block_cIdx = n * TblockDim * by + TblockDim * bx; // (blocklevel)
    // int thread_cIdx = block_cIdx + n * ty + tx;           // (threadlevel)

    //! Comment out the whole computation, at just load Q and store it in C
    //   int aBegin = k * TblockDim * by; // (blocklevel)
    //   int aEnd = aBegin + k; // (blocklevel)
    //   int aStep = TblockDim; // (blocklevel)
    //   int bBegin = TblockDim * bx; // (blocklevel)
    //   int bStep = TblockDim * n; // (blocklevel)

    //   // Csub is used to store the element of the block sub-matrix
    //   // that is computed by the thread
    //   // @max: Csub is also used to accumulate the result for one entry in
    //   the
    //   // output matrix C
    //   scalar_t Csub = dscalar_zero<scalar_t>();

    //   // Loop over all the sub-matrices of A and B
    //   // required to compute the block sub-matrix
    //   // @max: outer loop, progresses always in TblockDim steps and
    //   accumulates
    //   // the final values in Csub
    //   for (int a = aBegin, b = bBegin; a < aEnd; a += aStep, b += bStep) {
    //     __shared__ scalar_t As[TblockDim][TblockDim];
    //     __shared__ scalar_t Bs[TblockDim][TblockDim];

    //     As[ty][tx] = A[a + k * ty + tx]; // (threadlevel)
    //     Bs[ty][tx] = B[b + n * ty + tx]; // (threadlevel)
    //     // Synchronize to make sure the matrices are loaded
    //     __syncthreads();

    // #pragma unroll
    //     for (int i = 0; i < TblockDim; ++i) {
    //       if ((cx == 0) && (cy == 9)) {
    //         print_val("(ty,i) - As", ty, i, As[ty][i]);
    //         print_val("(ty,i) - Bs", ty, i, Bs[ty][i]);
    //       }
    //       Csub = add_g(Csub, mul_g(As[ty][i],
    //                                Bs[i][tx])); // (threadlevel): each
    //                                thread
    //                                             // operates on TblockDim
    //                                             elements
    //       if ((cx == 0) && (cy == 9)) {
    //         print_val("(cx,cy)-InLoop:Csub", cx, cy, Csub);
    //       }
    //     }
    //     if ((cx == 0) && (cy == 9)) {
    //       print_val("(cx,cy)-AfterLoop:Csub", cx, cy, Csub);
    //     }
    //     __syncthreads();
    //   }
    // matC[thread_cIdx] = Csub; // (threadlevel)
  }
}

template <typename scalar_t>
void kernel_dispatchers::qkvkernel_dispatch(scalar_t *matC, scalar_t *matQ,
                                            scalar_t *matK, scalar_t *matV,
                                            int batchSize, int numHeads,
                                            int seqLen, int dimHeads) {
  printf("B: %d, NH: %d, S: %d, DH: %d\n", batchSize, numHeads, seqLen,
         dimHeads);
  const int TblockDim = TBLOCK_DIM;  // matmul blockdim
  const int QblockDim = QTILE_DIM;   // blockdim for Q along seqLen dim
  const int KVblockDim = KVTILE_DIM; // blockdim for K&V along seqLen dim

  // kernel asserts
  if ((seqLen % QblockDim != 0) || (seqLen % KVblockDim != 0)) {
    printf("seqLen must be divisible by QblockDim and KVblockDim\n");
  }

  if (dimHeads >= HD_SIZE) {
    fprintf(stderr, "dimHeads must be smaller than HD_SIZE\n");
  }

  // determine the number of blocks and threads
  const dim3 blockDims(TblockDim, TblockDim);

  const dim3 gridDims(CEIL_DIV(KVblockDim, blockDims.x),
                      CEIL_DIV(QblockDim, blockDims.y));
  printf("blocksxy: %d-%d, threads: %d-%d\n", gridDims.x, gridDims.y,
         blockDims.x, blockDims.y);

  kernels::qkvkernel<scalar_t, TblockDim, QblockDim, KVblockDim>
      <<<gridDims, blockDims>>>(matC, matQ, matK, matV, batchSize, numHeads,
                                seqLen, dimHeads);

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
}

// this is needed to make sure that the compiler instantiates the template
template void kernel_dispatchers::qkvkernel_dispatch<__nv_bfloat16>(
    __nv_bfloat16 *matC, __nv_bfloat16 *matQ, __nv_bfloat16 *matK,
    __nv_bfloat16 *matV, int batchSize, int numHeads, int seqLen, int dimHeads);
template void kernel_dispatchers::qkvkernel_dispatch<__half>(
    __half *matC, __half *matQ, __half *matK, __half *matV, int batchSize,
    int numHeads, int seqLen, int dimHeads);

} // namespace vlstm