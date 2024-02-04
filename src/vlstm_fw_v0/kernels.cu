// Copyright JKU Linz 2023
// Author: Maximilian Beck

#include <cooperative_groups.h>
#include <cstdio>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_device_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
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

namespace cg = cooperative_groups;

namespace kernels {

template <typename scalar_t, int TblockDim, int QblockDim, int KVblockDim>
__global__ void vlstm_fw(scalar_t *matC, scalar_t *matQ, scalar_t *matK,
                         scalar_t *matV, int batchSize, int numHeads,
                         int seqLen, int dimHeads);

} // namespace kernels

////////////////////////////////////////////////////////////////////////////////////////

#define TBLOCK_DIM 4 // TblockDim: corresponds to BLOCK_DIM in matmul
#define KVTILE_DIM 8 // KVtileDim: TileDim for K&V along seqLen dim
// QTILE_DIM must be divisible by KVTILE_DIM and TBLOCK_DIM
#define QTILE_DIM 8 // QtileDim: TileDim for Q along seqLen dim

// shared memory must be aligned: depends on scalar_t (multiples of 4 should be
// fine for bf16, fp16 and fp32)
#define SHARED_MEM_PADDING 8 // SHARED_MEM_PADDING: padding for shared memory

// SMEMARRAY: access shared memory array
#define SMEMARRAY(array, stride, row, col)                                     \
  array[(row) * (stride + SHARED_MEM_PADDING) + (col)]

#define DEBUG 1
// #define DEBUG2 1
// #define DEBUG3 1
// #define DEBUG4 1
// #define DEBUG5 1

/**
Conventions:
// TODO add conventions

*/

/* vLSTM Forward Kernel v0 */

template <typename scalar_t, int TblockDim, int QtileDim, int KVtileDim>
__global__ void kernels::vlstm_fw(scalar_t *matC, scalar_t *matQ,
                                  scalar_t *matK, scalar_t *matV, int batchSize,
                                  int numHeads, int seqLen, int dimHeads) {
  // int tIdx = threadIdx.x + blockDim.x * threadIdx.y;
#ifdef DEBUG
  if ((blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0) &&
      (threadIdx.y == 0)) {
    printf("In Kernel: gdim.x: %d, gdim.y: %d, gdim.z: %d, bdim.x: %d, bdim.y: "
           "%d\n",
           gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y);
  }
#endif

#ifdef DEBUG
  if ((blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0) &&
      (threadIdx.y == 0)) {
    printf("In Kernel: QtileDim: %d, KVtileDim: %d, TblockDim:%d\n", QtileDim,
           KVtileDim, TblockDim);
  }
#endif
  cg::grid_group gridGroup = cg::this_grid();

  //! Shared Memory aka SRAM
  // the data in this shared memory is shared across all threads in a thread
  // block
  extern __shared__ float sbuf[]; // declare it as float and redefine it later

  //? for inputs
  // Note: keep in mind the memory is defined in a contiguous region.
  // One pointer has the full memory space until the next point is defined.
  // Therefore to read the size of a single shared memory array you need to
  // have a look at the offset for the next array.

  // qtile (QtileDim x dimHeads) in shared memory (padding for alignment)
  scalar_t *qTile = (scalar_t *)sbuf;
  // kTile and vTile (KVtileDim x dimHeads) in shared memory
  scalar_t *kTile =
      (scalar_t *)&qTile[QtileDim * (dimHeads + SHARED_MEM_PADDING)];
  scalar_t *vTile =
      (scalar_t *)&kTile[KVtileDim * (dimHeads + SHARED_MEM_PADDING)];

  //? for intermediate results
  // init cTile (QtileDim x KVTileDim) in shared memory for intermediate
  // result of QK^T
  scalar_t *cTile =
      (scalar_t *)&vTile[KVtileDim * (dimHeads + SHARED_MEM_PADDING)];
  // init result hTile (QTileDim x dimHeads) in shared memory
  scalar_t *hTile =
      (scalar_t *)&cTile[QtileDim * (KVtileDim + SHARED_MEM_PADDING)];
  // init dTile (QTileDim x KVTileDim) in shared memory for forget and input
  // gate matrix
  scalar_t *dTile =
      (scalar_t *)&hTile[QtileDim * (dimHeads + SHARED_MEM_PADDING)];

  //! PARALLELIZE ALONG BATCHSIZE * NUMHEADS (gridDim.x)
  const uint batchHeadStep = seqLen * dimHeads;
  const uint numBatchHeads = batchSize * numHeads;
  // End for looplevel 0:
  const uint batchHeadEnd = CEIL_DIV(numBatchHeads, gridDim.x);
  // looplevel 0: loop over batches and heads
  for (uint batchHeadIdx = 0; batchHeadIdx < batchHeadEnd; ++batchHeadIdx) {

    uint batchHeadGridXGlobalMemIdx =
        (batchHeadStep * gridDim.x) * batchHeadIdx + (batchHeadStep)*blockIdx.x;

#ifdef DEBUG5
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
      printf("B<%d,%d> batchHeadIdx: %d, batchHeadEnd: %d, "
             "batchHeadGridXGlobalMemIdx: "
             "%d\n",
             blockIdx.x, blockIdx.y, batchHeadIdx, batchHeadEnd,
             batchHeadGridXGlobalMemIdx);
    }
#endif

    //! PARALLELIZE ALONG SEQLEN (gridDim.y)
    // Ends for looplevel 1&2:
    const uint qTileEnd = CEIL_DIV(seqLen, QtileDim * gridDim.y);
    const uint kvTileEnd = CEIL_DIV(seqLen, KVtileDim);
    // looplevel 1: loop over Qtile blocks along seqLen dim
    for (uint qTileIdx = 0; qTileIdx < qTileEnd; ++qTileIdx) {

      // offset in Q matrix for qTile (global memory)
      const uint qTileGridXYGlobalMemIdx =
          batchHeadGridXGlobalMemIdx +
          (dimHeads * QtileDim * gridDim.y) * qTileIdx;
      const uint qTileBlockGlobalMemIdx =
          qTileGridXYGlobalMemIdx + (dimHeads * QtileDim) * blockIdx.y;

#ifdef DEBUG5
      if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        printf("B<%d,%d> qTileIdx: %d, qTileEnd: %d, "
               "qTileBlockGlobalMemIdx: "
               "%d, \n",
               blockIdx.x, blockIdx.y, qTileIdx, qTileEnd,
               qTileBlockGlobalMemIdx);
      }
#endif

#ifdef DEBUG2
      if ((blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0) &&
          (threadIdx.y == 0)) {
        printf("qTileIdx=%d: qTileEnd: %d, qTileGridXYGlobalMemIdx: %d, "
               "qTileBlockGlobalMemIdx: %d\n",
               qTileIdx, qTileEnd, qTileGridXYGlobalMemIdx,
               qTileBlockGlobalMemIdx);
      }
#endif
      //! qTile Loading
      // loops over rows (outer) and columns (inner) of qTile
      const uint qWarpTileYEnd = CEIL_DIV(QtileDim, blockDim.y);
      const uint qWarpTileXEnd = CEIL_DIV(dimHeads, blockDim.x);
      for (uint qWarpTileYIdx = 0; qWarpTileYIdx < qWarpTileYEnd;
           ++qWarpTileYIdx) {
#ifdef DEBUG2
        if ((blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0) &&
            (threadIdx.y == 0)) {
          printf("qWarpTileYIdx=%d: qWarpTileYEnd: %d, qWarpTileXEnd: %d\n",
                 qWarpTileYIdx, qWarpTileYEnd, qWarpTileXEnd);
        }
#endif
        for (uint qWarpTileXIdx = 0; qWarpTileXIdx < qWarpTileXEnd;
             ++qWarpTileXIdx) {
          //? qWarpTileIdxes
          //* shared memory:
          const uint qWarpTileThreadSharedMemYIdx =
              blockDim.y * qWarpTileYIdx + threadIdx.y;
          const uint qWarpTileThreadSharedMemXIdx =
              blockDim.x * qWarpTileXIdx + threadIdx.x;
          //* global memory:
          // left upper corner of qTileBlock in Q (global memory)
          const uint qWarpTileBlockGlobalMemIdx =
              qTileBlockGlobalMemIdx + (dimHeads * blockDim.y) * qWarpTileYIdx +
              blockDim.x * qWarpTileXIdx;
          const uint qWarpTileThreadGlobalMemIdx =
              qWarpTileBlockGlobalMemIdx + dimHeads * threadIdx.y + threadIdx.x;

          SMEMARRAY(qTile, dimHeads, qWarpTileThreadSharedMemYIdx,
                    qWarpTileThreadSharedMemXIdx) =
              matQ[qWarpTileThreadGlobalMemIdx];

#ifdef DEBUG3
          if ((blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0) &&
              (threadIdx.y == 0)) {
            printf("qTile[%d][%d] = %f\n", qWarpTileThreadSharedMemYIdx,
                   qWarpTileThreadSharedMemXIdx,
                   type2float(qTile[qWarpTileThreadSharedMemYIdx]
                                   [qWarpTileThreadSharedMemXIdx]));
          }
#endif
        }
      }
      __syncthreads(); // TODO: necessary?

      // looplevel 2: loop over KVtile blocks along seqLen dim
      for (uint kvTileIdx = 0; kvTileIdx < kvTileEnd; ++kvTileIdx) {

        // offset in K&V matrix for kTile & vTile (global memory)
        const uint kvTileBlockGlobalMemIdx =
            batchHeadGridXGlobalMemIdx + (dimHeads * KVtileDim) * kvTileIdx;

        //! kTile & vTile Loading
        // loops over rows (outer) and columns (inner) of kTile & vTile
        const uint kvWarpTileYEnd = CEIL_DIV(KVtileDim, blockDim.y);
        const uint kvWarpTileXEnd = CEIL_DIV(dimHeads, blockDim.x);
        for (uint kvWarpTileYIdx = 0; kvWarpTileYIdx < kvWarpTileYEnd;
             ++kvWarpTileYIdx) {
#ifdef DEBUG2
          if ((blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0) &&
              (threadIdx.y == 0)) {
            printf(
                "kvWarpTileYIdx=%d: kvWarpTileYEnd: %d, kvWarpTileXEnd: %d\n",
                kvWarpTileYIdx, kvWarpTileYEnd, kvWarpTileXEnd);
          }
#endif
          for (uint kvWarpTileXIdx = 0; kvWarpTileXIdx < kvWarpTileXEnd;
               ++kvWarpTileXIdx) {
            //? kvWarpTileIdxes
            //* shared memory:
            const uint kvWarpTileThreadSharedMemYIdx =
                blockDim.y * kvWarpTileYIdx + threadIdx.y;
            const uint kvWarpTileThreadSharedMemXIdx =
                blockDim.x * kvWarpTileXIdx + threadIdx.x;
            //* global memory:
            // left upper corner of kTileBlock in K (global memory)
            const uint kvWarpTileBlockGlobalMemIdx =
                kvTileBlockGlobalMemIdx +
                (dimHeads * blockDim.y) * kvWarpTileYIdx +
                blockDim.x * kvWarpTileXIdx;
            const uint kvWarpTileThreadGlobalMemIdx =
                kvWarpTileBlockGlobalMemIdx + dimHeads * threadIdx.y +
                threadIdx.x;

            SMEMARRAY(kTile, dimHeads, kvWarpTileThreadSharedMemYIdx,
                      kvWarpTileThreadSharedMemXIdx) =
                matK[kvWarpTileThreadGlobalMemIdx];
            SMEMARRAY(vTile, dimHeads, kvWarpTileThreadSharedMemYIdx,
                      kvWarpTileThreadSharedMemXIdx) =
                matV[kvWarpTileThreadGlobalMemIdx];
          }
        }
        __syncthreads();

        //! compute C = Q x K^T, i.e. fill cTile
        // (QtileDim,KVtileDim) = (QtileDim,dimHeads) x (dimHeads,KVtileDim)
        // loops over cTile rows (outer) and columns (inner)
        const uint sWarpTileYEnd = CEIL_DIV(QtileDim, blockDim.y);
        const uint sWarpTileXEnd = CEIL_DIV(KVtileDim, blockDim.x);
        for (uint sWarpTileYIdx = 0; sWarpTileYIdx < sWarpTileYEnd;
             ++sWarpTileYIdx) {
          for (uint sWarpTileXIdx = 0; sWarpTileXIdx < sWarpTileXEnd;
               ++sWarpTileXIdx) {
            //? sTileIdxes
            //* shared memory:
            const uint sWarpTileThreadSharedMemYIdx =
                blockDim.y * sWarpTileYIdx + threadIdx.y;
            const uint sWarpTileThreadSharedMemXIdx =
                blockDim.x * sWarpTileXIdx + threadIdx.x;

            // scalar_t qk_acc = dscalar_zero<scalar_t>();
            float qk_acc = 0.0f;
            for (uint i = 0; i < dimHeads; ++i) {
              qk_acc = add_g(qk_acc,
                             type2float(mul_g(
                                 SMEMARRAY(qTile, dimHeads,
                                           sWarpTileThreadSharedMemYIdx, i),
                                 SMEMARRAY(kTile, dimHeads,
                                           sWarpTileThreadSharedMemXIdx, i))));
#ifdef DEBUG4
              if ((blockIdx.x == 0) && (blockIdx.y == 0) &&
                  (threadIdx.x == 0) && (threadIdx.y == 3) &&
                  (sWarpTileXIdx == 0) && (kvTileIdx == 0) &&
                  (i == dimHeads - 1)) {
                printf("qTIdx=%d|kvTIdx=%d: qTile[%d][%d] = %f\n", qTileIdx,
                       kvTileIdx, sWarpTileThreadSharedMemYIdx, i,
                       type2float(qTile[sWarpTileThreadSharedMemYIdx][i]));
                printf("qTIdx=%d|kvTIdx=%d: kTile[%d][%d] = %f\n", qTileIdx,
                       kvTileIdx, sWarpTileThreadSharedMemXIdx, i,
                       type2float(kTile[sWarpTileThreadSharedMemXIdx][i]));
                printf("qTIdx=%d|kvTIdx=%d: cTile[%d][%d](%d) = %f\n", qTileIdx,
                       kvTileIdx, sWarpTileThreadSharedMemYIdx,
                       sWarpTileThreadSharedMemXIdx, i, type2float(qk_acc));
              }
#endif
            }
            SMEMARRAY(cTile, KVtileDim, sWarpTileThreadSharedMemYIdx,
                      sWarpTileThreadSharedMemXIdx) =
                float2type<scalar_t>(qk_acc);
            __syncthreads();
          }
        }

        //! compute H += C * V, i.e. fill hTile
        //! accumulate KVtiles to hTile
        // (QtileDim,dimHeads) = (QtileDim,KVtileDim) x (KVtileDim,dimHeads)
        // loops over hTile rows (outer) and columns (inner)
        const uint cWarpTileYEnd = CEIL_DIV(QtileDim, blockDim.y);
        const uint cWarpTileXEnd = CEIL_DIV(dimHeads, blockDim.x);
        for (uint cWarpTileYIdx = 0; cWarpTileYIdx < cWarpTileYEnd;
             ++cWarpTileYIdx) {
          for (uint cWarpTileXIdx = 0; cWarpTileXIdx < cWarpTileXEnd;
               ++cWarpTileXIdx) {

            //? cTileIdxes
            //* shared memory:
            const uint cWarpTileThreadSharedMemYIdx =
                blockDim.y * cWarpTileYIdx + threadIdx.y;
            const uint cWarpTileThreadSharedMemXIdx =
                blockDim.x * cWarpTileXIdx + threadIdx.x;

            // scalar_t sv_acc = dscalar_zero<scalar_t>();
            float sv_acc = 0.0f;
            for (uint i = 0; i < KVtileDim; ++i) {
              sv_acc = add_g(
                  sv_acc,
                  type2float(mul_g(SMEMARRAY(cTile, KVtileDim,
                                             cWarpTileThreadSharedMemYIdx, i),
                                   SMEMARRAY(vTile, dimHeads, i,
                                             cWarpTileThreadSharedMemXIdx))));
            }
            // accumulate over all KVtiles
            if (kvTileIdx == 0) {
              // we need to clear the hTile in first iteration
              SMEMARRAY(hTile, dimHeads, cWarpTileThreadSharedMemYIdx,
                        cWarpTileThreadSharedMemXIdx) =
                  float2type<scalar_t>(sv_acc);
            } else {
              SMEMARRAY(hTile, dimHeads, cWarpTileThreadSharedMemYIdx,
                        cWarpTileThreadSharedMemXIdx) =
                  add_g(SMEMARRAY(hTile, dimHeads, cWarpTileThreadSharedMemYIdx,
                                  cWarpTileThreadSharedMemXIdx),
                        float2type<scalar_t>(sv_acc));
            }
            __syncthreads();
          }
        }
      }

      //! write hTile to global memory (has the same memory index as qTile)
      // loops over hTile rows (outer) and columns (inner)
      const uint cWarpTileYEnd = CEIL_DIV(QtileDim, blockDim.y);
      const uint cWarpTileXEnd = CEIL_DIV(dimHeads, blockDim.x);
      for (uint cWarpTileYIdx = 0; cWarpTileYIdx < cWarpTileYEnd;
           ++cWarpTileYIdx) {
        for (uint cWarpTileXIdx = 0; cWarpTileXIdx < cWarpTileXEnd;
             ++cWarpTileXIdx) {

          //? cTileIdxes
          //* shared memory:
          const uint cWarpTileThreadSharedMemYIdx =
              blockDim.y * cWarpTileYIdx + threadIdx.y;
          const uint cWarpTileThreadSharedMemXIdx =
              blockDim.x * cWarpTileXIdx + threadIdx.x;
          //* global memory:
          // left upper corner of cWarpTileBlock in C (global memory)
          const uint cWarpTileBlockGlobalMemIdx =
              qTileBlockGlobalMemIdx + (dimHeads * blockDim.y) * cWarpTileYIdx +
              blockDim.x * cWarpTileXIdx;
          const uint cWarpTileThreadGlobalMemIdx =
              cWarpTileBlockGlobalMemIdx + dimHeads * threadIdx.y + threadIdx.x;

          matC[cWarpTileThreadGlobalMemIdx] =
              SMEMARRAY(hTile, dimHeads, cWarpTileThreadSharedMemYIdx,
                        cWarpTileThreadSharedMemXIdx);
        }
      }
      __syncthreads();
    }
  }
}

template <typename scalar_t>
void kernel_dispatchers::vlstm_fw_dispatch(scalar_t *matC, scalar_t *matQ,
                                           scalar_t *matK, scalar_t *matV,
                                           int batchSize, int numHeads,
                                           int seqLen, int dimHeads) {
  printf("B: %d, NH: %d, S: %d, DH: %d\n", batchSize, numHeads, seqLen,
         dimHeads);
  const int TblockDim = TBLOCK_DIM; // matmul blockdim
  const int QtileDim = QTILE_DIM;   // blockdim for Q along seqLen dim
  const int KVtileDim = KVTILE_DIM; // blockdim for K&V along seqLen dim

  // kernel asserts
  if ((seqLen % QtileDim != 0) || (seqLen % KVtileDim != 0)) {
    printf("seqLen must be divisible by QblockDim and KVblockDim\n");
  }

  // determine the number of blocks and threads
  const dim3 blockDims(TblockDim, TblockDim);

  // TODO: determine gridDims
  // Note @mbeck: should be dynamically allocated.
  // At first parallelize across batchSize and numHeads.
  // If more streaming multiprocessors available, parallelize across seqLen.
  //! NOTE: for now we only parallelize across batchSize and numHeads
  // TODO Need to dynamically check how many blocks we can launch
  // TODO add check if batchSize*numHeads exceeds max gridDim.x

  const dim3 gridDims(batchSize * numHeads, 2);
  // const dim3 gridDims(1, 1);

  //! calculate dynamic shared memory size
  // TODO understand how memory padding works!
  // Why at innermost dim? Because memory is organized consecutively
  // we are storing the following tiles in shared memory:
  // - Input tiles: qTile, vTile, kTile -> (QtileDim, dimHeads +
  // SHARED_MEM_PADDING)
  // TODO from here add input & forgetgate tiles
  // - Intermediate result tile: cTile, dTile -> (QtileDim, KVtileDim +
  // SHARED_MEM_PADDING)
  // - Output tile: hTile -> (QtileDim, dimHeads + SHARED_MEM_PADDING)

  const uint qkvhTileSharedMemSize =
      sizeof(scalar_t) * QtileDim * (dimHeads + SHARED_MEM_PADDING);
  const uint cdTileSharedMemSize =
      sizeof(scalar_t) * QtileDim * (KVtileDim + SHARED_MEM_PADDING);

  // Input/Output tiles: 4x for qTile, vTile, kTile, hTile
  // Intermediate tiles: 2x for cTile, dTile
  const uint sharedMemorySize =
      4 * qkvhTileSharedMemSize + 2 * cdTileSharedMemSize;

  printf("blocksxy: %d-%d, threads: %d-%d, shared_mem in bytes: %d\n",
         gridDims.x, gridDims.y, blockDims.x, blockDims.y, sharedMemorySize);
  // cudaSetDevice(0);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  auto kernel = kernels::vlstm_fw<scalar_t, TblockDim, QtileDim, KVtileDim>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       sharedMemorySize);

  // define void* pointers to the kernel arguments
  void *kernelArgs[] = {(void *)&matC,   (void *)&matQ,      (void *)&matK,
                        (void *)&matV,   (void *)&batchSize, (void *)&numHeads,
                        (void *)&seqLen, (void *)&dimHeads};

  cudaLaunchCooperativeKernel((void *)kernel, gridDims, blockDims, kernelArgs,
                              sharedMemorySize, stream);

  gpuErrchk(cudaPeekAtLastError());

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  gpuErrchk(cudaDeviceSynchronize());

  // gpuErrchk(cudaPeekAtLastError());
  // gpuErrchk(cudaDeviceSynchronize());
}

// this is needed to make sure that the compiler instantiates the template
template void kernel_dispatchers::vlstm_fw_dispatch<__nv_bfloat16>(
    __nv_bfloat16 *matC, __nv_bfloat16 *matQ, __nv_bfloat16 *matK,
    __nv_bfloat16 *matV, int batchSize, int numHeads, int seqLen, int dimHeads);
template void kernel_dispatchers::vlstm_fw_dispatch<__half>(
    __half *matC, __half *matQ, __half *matK, __half *matV, int batchSize,
    int numHeads, int seqLen, int dimHeads);
template void kernel_dispatchers::vlstm_fw_dispatch<float>(
    float *matC, float *matQ, float *matK, float *matV, int batchSize,
    int numHeads, int seqLen, int dimHeads);

} // namespace vlstm