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
__global__ void qkvkernel(scalar_t *matC, scalar_t *matQ, scalar_t *matK,
                          scalar_t *matV, int batchSize, int numHeads,
                          int seqLen, int dimHeads);

} // namespace kernels

////////////////////////////////////////////////////////////////////////////////////////

#define TBLOCK_DIM 4 // TblockDim: corresponds to BLOCK_DIM in matmul
#define KVTILE_DIM 8 // KVtileDim: TileDim for K&V along seqLen dim
// QTILE_DIM must be divisible by KVTILE_DIM and TBLOCK_DIM
#define QTILE_DIM 8 // QtileDim: TileDim for Q along seqLen dim

// TODO use dynamic shared memory!
#define HD_SIZE                                                                \
  64 // HD_SIZE: size of the allocated shared memory for the hidden dim

#define DEBUG 1
// #define DEBUG2 1
// #define DEBUG3 1
#define DEBUG4 1

/**
Conventions:

..MemIdx - indices into global memory
..tileX / ..tileY - indices into shared memory tiles

*/

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

#ifdef DEBUG
  if ((blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0) &&
      (threadIdx.y == 0)) {
    printf("In Kernel: QtileDim: %d, KVtileDim: %d, TblockDim:%d\n", QtileDim,
           KVtileDim, TblockDim);
  }
#endif
  cg::grid_group gridGroup = cg::this_grid();

  // Assign threads to x-y-coordinates for accessing shared memory tiles
  const uint tileX = blockIdx.x * blockDim.x + threadIdx.x;
  const uint tileY = blockIdx.y * blockDim.y + threadIdx.y;

  // TODO Most outer loop: parallelize along gridDim.x
  // Most outer loop: Loop over batchSize * numHeads (can be parallelized later
  // along gridDim.y)
  const uint batchHeadStep = seqLen * dimHeads;
  const uint batchHeadEnd = batchSize * numHeads * batchHeadStep;
  //! Note: batchHeadMemIdx indices into global memory
  for (uint batchHeadMemIdx = 0; batchHeadMemIdx < batchHeadEnd;
       batchHeadMemIdx += batchHeadStep) {

    // Ends for looplevel 1&2:
    const uint qTileEnd = CEIL_DIV(seqLen, QtileDim);
    const uint kvTileEnd = CEIL_DIV(seqLen, KVtileDim);
    // TODO add looplevel 0: parallelize along gridDim.y
    // looplevel 1: loop over Qtile blocks along seqLen dim
    // Note: qTileIdx does not index into global memory
    for (uint qTileIdx = 0; qTileIdx < qTileEnd; ++qTileIdx) {
      // offset in Q matrix for qTile (global memory)
      const uint qTileGridGlobalMemIdx =
          batchHeadMemIdx + (dimHeads * QtileDim * gridDim.y) * qTileIdx;
      const uint qTileBlockGlobalMemIdx =
          qTileGridGlobalMemIdx + dimHeads * blockDim.y * blockIdx.y;

#ifdef DEBUG2
      if ((blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0) &&
          (threadIdx.y == 0)) {
        printf("qTileIdx=%d: qTileEnd: %d, qTileGridGlobalMemIdx: %d, "
               "qTileBlockGlobalMemIdx: %d\n",
               qTileIdx, qTileEnd, qTileGridGlobalMemIdx,
               qTileBlockGlobalMemIdx);
      }
#endif
      //! qTile Loading
      // init qTile in shared memory
      // TODO use dynamic shared memory!
      __shared__ scalar_t qTile[QtileDim][HD_SIZE];
      // initialize result cTile in shared memory
      __shared__ scalar_t cTile[QtileDim][HD_SIZE];

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

          qTile[qWarpTileThreadSharedMemYIdx][qWarpTileThreadSharedMemXIdx] =
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
      __syncthreads();
      //! DEBUG:
      __shared__ scalar_t kTile[KVtileDim][HD_SIZE];
      __shared__ scalar_t vTile[KVtileDim][HD_SIZE];

      // looplevel 2: loop over KVtile blocks along seqLen dim
      for (uint kvTileIdx = 0; kvTileIdx < kvTileEnd; ++kvTileIdx) {

        // offset in K&V matrix for kTile & vTile (global memory)
        const uint kvTileBlockGlobalMemIdx =
            batchHeadMemIdx + (dimHeads * KVtileDim) * kvTileIdx;

        //! kTile & vTile Loading
        // init kTile and vTile in shared memory
        // TODO use dynamic shared memory!
        // __shared__ scalar_t kTile[KVtileDim][HD_SIZE];
        // __shared__ scalar_t vTile[KVtileDim][HD_SIZE];

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

            kTile[kvWarpTileThreadSharedMemYIdx]
                 [kvWarpTileThreadSharedMemXIdx] =
                     matK[kvWarpTileThreadGlobalMemIdx];
            vTile[kvWarpTileThreadSharedMemYIdx]
                 [kvWarpTileThreadSharedMemXIdx] =
                     matV[kvWarpTileThreadGlobalMemIdx];
          }
        }
        __syncthreads();

        //! compute S = Q x K^T, i.e. fill sTile
        // (QtileDim,KVtileDim) = (QtileDim,dimHeads) x (dimHeads,KVtileDim)

        // init sTile (QtileDim x KVTileDim) in shared memory for intermediate
        // result of QK^T
        __shared__ scalar_t sTile[QtileDim][KVtileDim];

        // loops over sTile rows (outer) and columns (inner)
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

            scalar_t qk_acc = dscalar_zero<scalar_t>();
            for (uint i = 0; i < dimHeads; ++i) {
              qk_acc =
                  add_g(qk_acc, mul_g(qTile[sWarpTileThreadSharedMemYIdx][i],
                                      kTile[sWarpTileThreadSharedMemXIdx][i]));
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
                printf("qTIdx=%d|kvTIdx=%d: sTile[%d][%d](%d) = %f\n", qTileIdx,
                       kvTileIdx, sWarpTileThreadSharedMemYIdx,
                       sWarpTileThreadSharedMemXIdx, i, type2float(qk_acc));
              }
#endif
            }
            sTile[sWarpTileThreadSharedMemYIdx][sWarpTileThreadSharedMemXIdx] =
                qk_acc;
            __syncthreads();
          }
        }

        //! compute C += S * V, i.e. fill cTile
        //! accumulate KVtiles to cTile
        // (QtileDim,dimHeads) = (QtileDim,KVtileDim) x (KVtileDim,dimHeads)

        // loops over cTile rows (outer) and columns (inner)
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

            scalar_t sv_acc = dscalar_zero<scalar_t>();
            for (uint i = 0; i < KVtileDim; ++i) {
              sv_acc =
                  add_g(sv_acc, mul_g(sTile[cWarpTileThreadSharedMemYIdx][i],
                                      vTile[i][cWarpTileThreadSharedMemXIdx]));
            }
            // accumulate over all KVtiles
            cTile[cWarpTileThreadSharedMemYIdx][cWarpTileThreadSharedMemXIdx] =
                add_g(cTile[cWarpTileThreadSharedMemYIdx]
                           [cWarpTileThreadSharedMemXIdx],
                      sv_acc);
            __syncthreads();
          }
        }
      }

      //! write cTile to global memory (has the same memory index as qTile)
      // loops over cTile rows (outer) and columns (inner)
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
              cTile[cWarpTileThreadSharedMemYIdx][cWarpTileThreadSharedMemXIdx];
          // matC[cTileThreadGlobalMemIdx] =
          //     kTile[cTileThreadSharedMemYIdx][cTileThreadSharedMemXIdx];
        }
      }
      __syncthreads();
    }
  }
}

template <typename scalar_t>
void kernel_dispatchers::qkvkernel_dispatch(scalar_t *matC, scalar_t *matQ,
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

  if (dimHeads >= HD_SIZE) {
    fprintf(stderr, "dimHeads must be smaller than HD_SIZE\n");
  }

  // determine the number of blocks and threads
  const dim3 blockDims(TblockDim, TblockDim);

  // const dim3 gridDims(CEIL_DIV(dimHeads, blockDims.x),
  //                     CEIL_DIV(QtileDim, blockDims.y));
  const dim3 gridDims(1, 1);
  printf("blocksxy: %d-%d, threads: %d-%d\n", gridDims.x, gridDims.y,
         blockDims.x, blockDims.y);

  // TODO calculate dynamic shared memory size

  cudaSetDevice(0);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  auto kernel = kernels::qkvkernel<scalar_t, TblockDim, QtileDim, KVtileDim>;
  // cudaFuncSetAttribute(kernel,
  // cudaFuncAttributePreferredSharedMemoryCarveout,
  //                      cudaSharedmemCarveoutMaxShared);
  // TODO dynamic shared memory
  // cudaFuncSetAttribute(kernel,
  // cudaFuncAttributeMaxDynamicSharedMemorySize,
  //                      sharedMemorySize);

  // define void* pointers to the kernel arguments
  void *kernelArgs[] = {(void *)&matC,   (void *)&matQ,      (void *)&matK,
                        (void *)&matV,   (void *)&batchSize, (void *)&numHeads,
                        (void *)&seqLen, (void *)&dimHeads};

  cudaLaunchCooperativeKernel((void *)kernel, gridDims, blockDims, kernelArgs,
                              0, stream);

  // kernels::qkvkernel<scalar_t, TblockDim, QtileDim, KVtileDim>
  //     <<<gridDims, blockDims>>>(matC, matQ, matK, matV, batchSize,
  //     numHeads,
  //                               seqLen, dimHeads);
  gpuErrchk(cudaPeekAtLastError());

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  gpuErrchk(cudaDeviceSynchronize());

  // gpuErrchk(cudaPeekAtLastError());
  // gpuErrchk(cudaDeviceSynchronize());
}

// this is needed to make sure that the compiler instantiates the template
template void kernel_dispatchers::qkvkernel_dispatch<__nv_bfloat16>(
    __nv_bfloat16 *matC, __nv_bfloat16 *matQ, __nv_bfloat16 *matK,
    __nv_bfloat16 *matV, int batchSize, int numHeads, int seqLen, int dimHeads);
template void kernel_dispatchers::qkvkernel_dispatch<__half>(
    __half *matC, __half *matQ, __half *matK, __half *matV, int batchSize,
    int numHeads, int seqLen, int dimHeads);

} // namespace vlstm