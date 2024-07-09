
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../util/cuda_errorcheck.h"
#include "../util/inline_ops.cuh"
#include "../util/inline_print.cuh"
#include "../util/support.h"
#include "kernel_dispatchers.h"

#include <assert.h>
#include <cuda.h>
#include <cuda/pipeline>
#include <cuda_bf16.h>
#include <mma.h>
#include <stdio.h>

// helper functions and utilities to work with CUDA
#include "helper_cuda.h"
// #include <helper_functions.h>

/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// CUDA sample demonstrating a __nv_bfloat16 (E8M7) GEMM computation using the
// Warp Matrix Multiply and Accumulate API introduced in CUDA 11.0.

// In this program, the compute_gemm kernel computes the result of a matrix
// multiplication and addition: D = alpha * A * B + beta * C. The dimensions of
// both C and D matrices are M_GLOBAL x N_GLOBAL. The A matrix is M_GLOBAL x
// K_GLOBAL (row-major), the B matrix is K_GLOBAL x N_GLOBAL (column-major). In
// that kernel, each CTA computes one 128 x 128 tile of the resulting matrix per
// iteration. When the tile is computed, the CTA stores it to the global memory
// and begins a new iteration, selecting a new 128 x 128 tile to compute.
// Each CTA consists of eight warps. For the 128 x 128 tile, each warp computes
// eight 16 x 16 subtiles, organized in a 2 x 4 two-dimensional array. Warps
// compute the 16 x 16 subtiles using nvcuda::wmma::mma_sync operations by
// moving through the K_GLOBAL dimension of the A and B matrices and
// accumulating the intermediate result in the local thread state.

// There are a number of simple optimizations used in the algorithm:
// - The CTA copies the 128 x 128 tile of the C matrix from the global memory to
//   shared memory. After that is done, each warp loads the C matrix fragments
//   from shared memory, thus avoiding a random global memory access.
// - On each internal iteration, the CTA copies a portion of the A and B
// matrices from
//   global memory to shared memory. After that, all warps in the CTA reuse the
//   A and B data from shared memory, thus reducing the number of data copies
//   from global memory.
// - The portions of the A and B matrices are stored in shared memory with an
// additional
//   padding (skew) to reduce the number of shared memory access bank conflicts.
//   (See a detailed explanation near the SKEW_BF16 macro definition.)
// - When the CTA finishes computing the tiles of the resulting matrix, each
// warp stores
//   its subtiles to shared memory. The CTA then copies the shared memory
//   contents to global memory, again avoiding redundant random global memory
//   accesses.
// - Note that the CTA tile size is chosen to maximize the GPU register
// utilization,
//   but carefully enough to avoid local memory use.

// Externally configurable parameters.

#ifndef CPU_DEBUG
// Set this to 1 to verify the correctness of the GPU-computed matrix.
#define CPU_DEBUG 0
#endif

#ifndef SHARED_MEMORY_LIMIT_64K
// Set this to 0 to use more than 64 Kb of shared memory to cache data, to
// improve the performance of the computations on GPU.
// Note that you need a GPU that can have more than 64 Kb of shared memory
// per multiprocessor.
#define SHARED_MEMORY_LIMIT_64K 0
#endif

// GPU configuration.

#define WARP_SIZE 32

// MMA matrix tile dimensions.

#define M 16
#define N 16
#define K 16

// GEMM configuration.

#define M_TILES 512
#define N_TILES 512
#define K_TILES 512

#define M_GLOBAL (M * M_TILES)
#define N_GLOBAL (N * N_TILES)
#define K_GLOBAL (K * K_TILES)

#define C_LAYOUT wmma::mem_row_major

// Implementation constants.

#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#if SHARED_MEMORY_LIMIT_64K
// With only 64 Kb shared memory available, we can fit two 8-tile chunks of
// the A and B matrix data, that is (M = 16) * (K = 16) * 8 * (CHUNK_K = 8)
// * sizeof(__nv_bfloat16) = 32 Kb each.
// (i.e. two 8x8 arrays of tiles of 16x16 __nv_bfloat16-typed elements per CTA).
// But we cannot account the 8 Kb total skew overhead, without which the
// performance would be severely impacted. So we choose to reduce the chunk size
// in half, i.e. the amount of A and B matrix data we cache in shared memory.
// Accordingly, this doubles the number of outer iterations across the global K
// dimension, which only slightly impacts the performance.
#define CHUNK_K 4
#else
#define CHUNK_K 8
#endif

#define CHUNK_LINE_BYTES (CHUNK_K * K * sizeof(__nv_bfloat16))
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define GLOBAL_MEM_STRIDE N_GLOBAL

#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)

// The macro below is used to shift rows of the A matrix and columns of the B
// matrix in shared memory to minimize possible bank conflicts. Before
// performing the nvcuda::wmma::mma_sync operation, the warp must load the
// matrix data using the nvcuda::wmma::load_matrix_sync operation. Although the
// memory access pattern is not specified for that function, each lane in the
// warp can read one or multiple matrix elements from different matrix rows or
// columns. For shared memory, such access can result in bank conflicts if
// different rows / columns of the matrix map to the same bank. By shifting each
// row and column by a few bytes, we make sure that they map to different banks,
// thus reducing the number of possible bank conflicts. The number of 16
// two-byte "__nv_bfloat16" elements is chosen as the minimum possible shift
// because we must keep each row and column 256-bit aligned, as required by
// nvcuda::wmma::load_matrix_sync.
#define SKEW_BF16 16

//! Only for readability purposes. Uncomment
#define __CUDA_ARCH__ 800

#define checkKernelErrors(expr)                                                \
  do {                                                                         \
    expr;                                                                      \
                                                                               \
    cudaError_t __err = cudaGetLastError();                                    \
    if (__err != cudaSuccess) {                                                \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr,                    \
             cudaGetErrorString(__err));                                       \
      abort();                                                                 \
    }                                                                          \
  } while (0)

// enum kernels {
//   bf16mma_shmem_gemm_async_copy =
//       0, // __nv_bfloat16 MMA shmem using kernel with async_copy
//   bf16mma_shmem_gemm = 1, // __nv_bfloat16 MMA shmem using kernel normal copy
//                           // (without async_copy).
//   simple_bf16mma_gemm = 2 // __nv_bfloat16 MMA non-shmem using simple kernel.
// };

// const char *kernelNames[] = {"compute_bf16gemm_async_copy",
// "compute_bf16gemm",
//                              "simple_wmma_bf16gemm"};

using namespace nvcuda;

namespace vlstm {

// namespace kernels {

// /*MatrixMul kernel does matrix multiplication from the NVIDA cuda_samples
//  * repo.*/
// template <typename scalar_t, int BLOCKSIZE>
// __global__ void mmkernel(scalar_t *matC, scalar_t *matA, scalar_t *matB, int
// m,
//                          int n, int k);

// } // namespace kernels

__global__ void mmkernel(const __nv_bfloat16 *A, const __nv_bfloat16 *B,
                         const float *C, float *D, float alpha, float beta) {
#if __CUDA_ARCH__ >= 800
  extern __shared__ __nv_bfloat16 shmem[][CHUNK_K * K + SKEW_BF16];

  // Warp and lane identification.
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const unsigned int laneId = threadIdx.x % WARP_SIZE;

  // Offset in shared memory from which the B matrix is stored.
  const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

  // This pointer is used to access the C and D matrix tiles this warp computes.
  float *shmem_warp_tile_ptr =
      (float *)&shmem[0][0] +
      (warpId / BLOCK_ROW_WARPS) * SHMEM_STRIDE * N * BLOCK_ROW_WARPS +
      (warpId % BLOCK_ROW_WARPS) * SHMEM_OFFSET;

  // This pointer is used to stream the C and D matrices block-wide tile to and
  // from shared memory.
  float *shmem_warp_stream_ptr =
      (float *)&shmem[0][0] + warpId * SHMEM_STRIDE * N;

  // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
  // each tile computation. Technically this is not generally correct (may
  // result in a loss of precision). Zero still needs to be specially handled
  // though.
  beta /= alpha;

  // Each CTA slides along the 128 x 128 tiles from the top left corner of the
  // matrix to the right and down, and selects the next tile to compute. Once
  // there's no such tile, all warps in this CTA exit.
  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int block_tile_i =
        ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
    const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

    // Stop when there are no more D matrix tiles to compute in this CTA.
    if (block_tile_i >= M_TILES) {
      break;
    }

    // This warp's pointer to the C matrix data to copy memory from to shared
    // memory.
    const size_t gmem_idx =
        (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;
    const float *src_gmem_warp_stream_ptr = &C[gmem_idx];

    // Stream multiple C tiles to shared memory.
#pragma unroll
    for (int i = 0; i < N; i++) {
      *((int4 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) = *(
          (int4 *)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId);
    }

    __syncthreads();

    // These fragments will accumulate the result of A and B matrix fragment
    // multiplications along the K_GLOBAL dimension.
    wmma::fragment<wmma::accumulator, M, N, K, float> c[WARP_COL_TILES]
                                                       [WARP_ROW_TILES];

    // Load the C matrix tiles into fragments from shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
        const float *tile_ptr =
            shmem_warp_tile_ptr + i * SHMEM_STRIDE * N + j * N;

        wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
      }
    }

    __syncthreads();

    // Scale the C matrix.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
        for (int t = 0; t < c[i][j].num_elements; t++) {
          c[i][j].x[t] *= beta;
        }
      }
    }

    // Select what warp copies what matrix to shared memory.
    // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
    const __nv_bfloat16 *warp_ptr =
        (warpId < (WARPS_PER_BLOCK / 2))
            ? (&A[block_tile_i * M * K_GLOBAL] +
               M * K_GLOBAL * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
            : (&B[block_tile_j * N * K_GLOBAL] +
               N * K_GLOBAL * (warpId % (WARPS_PER_BLOCK / 2)) * 2);

    // Go through the global K dimension by a fixed step at a time.
#pragma unroll
    for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
      // Copy slices of the A and B matrices to shared memory.
      // The first half of the warps in the CTA copy the A matrix, the rest copy
      // the B matrix.
      size_t shmem_idx =
          warpId < (WARPS_PER_BLOCK / 2)
              ? (M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
              : (N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

      // First half of the warp copies the first row / column of the matrix,
      // the second half of the warp copies the next.
      const __nv_bfloat16 *lane_ptr =
          (warp_ptr + tile_k * K + (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL);

      // Shift the second half of the warp to the next row / column in the
      // shared memory.
      shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

#pragma unroll
      for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2;
           i++) {
        // Copy 16 bytes at once in each lane.
        *((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
            *((int4 *)lane_ptr + (laneId % CHUNK_COPY_LINE_LANES));

        // Advance the global memory pointer and the shared memory index.
        lane_ptr = lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP;
        shmem_idx += CHUNK_COPY_LINES_PER_WARP;
      }

      __syncthreads();

      // Compute a grid of C matrix tiles in each warp.
#pragma unroll
      for (int k_step = 0; k_step < CHUNK_K; k_step++) {
        wmma::fragment<wmma::matrix_a, M, N, K, __nv_bfloat16, wmma::row_major>
            a[WARP_COL_TILES];
        wmma::fragment<wmma::matrix_b, M, N, K, __nv_bfloat16, wmma::col_major>
            b[WARP_ROW_TILES];

#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
          size_t shmem_idx_a =
              (warpId / BLOCK_ROW_WARPS) * M * BLOCK_ROW_WARPS + (i * M);
          const __nv_bfloat16 *tile_ptr = &shmem[shmem_idx_a][k_step * K];

          wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_BF16);

#pragma unroll
          for (int j = 0; j < WARP_ROW_TILES; j++) {
            if (i == 0) {
              // Load the B matrix fragment once, because it is going to be
              // reused against the other A matrix fragments.
              size_t shmem_idx_b = shmem_idx_b_off +
                                   (WARP_ROW_TILES * N) * (warpId % 2) +
                                   (j * N);
              const __nv_bfloat16 *tile_ptr = &shmem[shmem_idx_b][k_step * K];

              wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_BF16);
            }

            wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
          }
        }
      }

      __syncthreads();
    }

    // Store the D fragments to shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
        // Uniform, point-wise transformations of ALL fragment elements by ALL
        // threads in the warp are well-defined even though element indices
        // within fragment storage are not defined.
        for (int t = 0; t < c[i][j].num_elements; t++)
          c[i][j].x[t] *= alpha;

        float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

        wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
      }
    }

    __syncthreads();

    // Now that shared memory contains all the D tiles, stream them to global
    // memory.
    float *dst_gmem_warp_stream_ptr = &D[gmem_idx];

#pragma unroll
    for (int i = 0; i < N; i++) {
      *((float4 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
          *((float4 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
    }

    __syncthreads();
  }
#endif
}

/*
A: (m x k)
B: (k x n)
C: (m x n)
*/
template <typename scalar_t>
void kernel_dispatchers::mmkernel_dispatch(float *matD, scalar_t *matA,
                                           scalar_t *matB, float *matC, int m,
                                           int n, int k) {
  // printf("m: %d, n: %d, k: %d\n", m, n, k);

  enum {
    // Compute the right amount of shared memory to request.
    // We need shared memory to hold per-CTA C and D matrix tiles, and to cache
    // per-CTA chunks
    // of the A and B matrices. Therefore, the right amount to request is the
    // maximum of those
    // two numbers.
    SHMEM_SZ = MAX(sizeof(__nv_bfloat16) * (BLOCK_COL_TILES * M) *
                       (CHUNK_K * K + SKEW_BF16) * 2,
                   M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N *
                       (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float))
  };
  uint shmem_sz_1 = sizeof(__nv_bfloat16) * (BLOCK_COL_TILES * M) *
                    (CHUNK_K * K + SKEW_BF16) * 2;
  uint shmem_sz_2 = M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N *
                    (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float);
  printf("Required shared memory size: %d bytes = MAX(%d, %d)\n", SHMEM_SZ,
         shmem_sz_1, shmem_sz_2);
  printf("Required shared memory size: %lu Kb\n", SHMEM_SZ / 1024UL);

  printf("#bytes: sizeof(int4)= %lu, sizeof(float4)= %lu, "
         "sizeof(__nv_bfloat16)= %lu\n",
         sizeof(int4), sizeof(float4), sizeof(__nv_bfloat16));
  printf("CHUNK_K=%d, CHUNK_LINE_BYTES=%lu, WARP_COPY_BYTES=%lu, "
         "CHUNK_COPY_LINES_PER_WARP=%lu, CHUNK_COPY_LINE_LANES=%lu\n",
         CHUNK_K, CHUNK_LINE_BYTES, WARP_COPY_BYTES, CHUNK_COPY_LINES_PER_WARP,
         CHUNK_COPY_LINE_LANES);
  printf("BLOCK_ROW_TILES=%d, BLOCK_COL_TILES=%d, BLOCK_ROW_WARPS=%d, "
         "BLOCK_COL_WARPS=%d\n",
         BLOCK_ROW_TILES, BLOCK_COL_TILES, BLOCK_ROW_WARPS, BLOCK_COL_WARPS);
  printf("WARP_ROW_TILES=%d, WARP_COL_TILES=%d, WARP_SIZE=%d, "
         "WARPS_PER_BLOCK=%d\n",
         WARP_ROW_TILES, WARP_COL_TILES, WARP_SIZE, WARPS_PER_BLOCK);
  printf("SHMEM_STRIDE=%d, SHMEM_OFFSET=%d\n", SHMEM_STRIDE, SHMEM_OFFSET);
  const float alpha = 1.1f;
  const float beta = 1.2f;

  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));

  checkCudaErrors(cudaFuncSetAttribute(
      mmkernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));

  checkKernelErrors(
      (mmkernel<<<deviceProp.multiProcessorCount * 2, THREADS_PER_BLOCK,
                  SHMEM_SZ>>>(matA, matB, matC, matD, alpha, beta)));

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
}

template void kernel_dispatchers::mmkernel_dispatch<__nv_bfloat16>(
    float *matD, __nv_bfloat16 *matA, __nv_bfloat16 *matB, float *matC, int m,
    int n, int k);
// template void kernel_dispatchers::mmkernel_dispatch<__half>(__half *matC,
//                                                             __half *matA,
//                                                             __half *matB, int
//                                                             m, int n, int k);

} // namespace vlstm