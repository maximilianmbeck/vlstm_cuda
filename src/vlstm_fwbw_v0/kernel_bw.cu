// Copyright JKU Linz 2024
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
#include <math_constants.h>
#include <sys/types.h>

#include "../util/cuda_errorcheck.h"
#include "../util/inline_ops.cuh"
#include "../util/inline_print.cuh"
#include "../util/support.h"
#include "kernel_dispatchers.h"

#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))
#define FLOOR_DIV(a, b) ((a) / (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

namespace vlstm {

namespace cg = cooperative_groups;

namespace kernels {

template <typename scalar_t, int TblockDim, int QblockDim, int KVblockDim>
__global__ void vlstm_bw(scalar_t *deltaQ, scalar_t *deltaK, scalar_t *deltaV,
                         scalar_t *deltaIGatePreact, scalar_t *deltaFGatePreact,
                         scalar_t *deltaH, scalar_t *matQ, scalar_t *matK,
                         scalar_t *matV, scalar_t *iGatePreact,
                         scalar_t *fGatePreact, scalar_t *vecN, scalar_t *vecM,
                         int batchSize, int numHeads, int seqLen, int dimHeads);

} // namespace kernels

////////////////////////////////////////////////////////////////////////////////////////

#define TBLOCK_DIM 4 // TblockDim: corresponds to BLOCK_DIM in matmul
#define KVTILE_DIM 8 // KVtileDim: TileDim for K&V along seqLen dim
// QTILE_DIM must be divisible by KVTILE_DIM and TBLOCK_DIM,
// KVTILE_DIM <= QTILE_DIM
#define QTILE_DIM 8 // QtileDim: TileDim for Q along seqLen dim

// shared memory must be aligned: depends on scalar_t (multiples of 4 should be
// fine for bf16, fp16 and fp32)
#define SHARED_MEM_PADDING 8 // SHARED_MEM_PADDING: padding for shared memory

// SMEMARRAY: access shared memory array (2D)
#define SMEMARRAY(array, stride, row, col)                                     \
  array[(row) * (stride + SHARED_MEM_PADDING) + (col)]
// SMEMVECTOR: access shared memory vector (1D)
#define SMEMVECTOR(array, idx) array[(idx) * (1 + SHARED_MEM_PADDING)]

#define DEBUG 1
// #define DEBUG2 1
// #define DEBUG3 1
// #define DEBUG4 1
// #define DEBUG5 1
// #define DEBUG6 1 // print fTileCol vals
// #define DEBUG7 1
// #define DEBUG8 1
// #define DEBUG9 1
// #define DEBUG10 1
// #define DEBUG11 1
// #define DEBUG12 1

/**
Conventions:
- chunk: A 1D vector in shared memory
- tile: A 2D matrix in shared memory
*/

/* vLSTM Backward Kernel v0 */

template <typename scalar_t, int TblockDim, int QtileDim, int KVtileDim>
__global__ void
kernels::vlstm_bw(scalar_t *deltaQ, scalar_t *deltaK, scalar_t *deltaV,
                  scalar_t *deltaIGatePreact, scalar_t *deltaFGatePreact,
                  scalar_t *deltaH, scalar_t *matQ, scalar_t *matK,
                  scalar_t *matV, scalar_t *iGatePreact, scalar_t *fGatePreact,
                  scalar_t *vecN, scalar_t *vecM, int batchSize, int numHeads,
                  int seqLen, int dimHeads) {
  // int tIdx = threadIdx.x + blockDim.x * threadIdx.y;
#ifdef DEBUG
  if ((blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0) &&
      (threadIdx.y == 0)) {
    printf(
        "In BW-Kernel: gdim.x: %d, gdim.y: %d, gdim.z: %d, bdim.x: %d, bdim.y: "
        "%d\n",
        gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y);
  }
#endif

#ifdef DEBUG
  if ((blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0) &&
      (threadIdx.y == 0)) {
    printf("In BW-Kernel: QtileDim: %d, KVtileDim: %d, TblockDim:%d\n",
           QtileDim, KVtileDim, TblockDim);
  }
#endif
  cg::grid_group gridGroup = cg::this_grid();

  //! Shared Memory aka SRAM
  // the data in this shared memory is shared across all threads in a thread
  // block
  extern __shared__ float sbuf[]; // declare it as float and redefine it later
  // Note: keep in mind the memory is defined in a contiguous region.
  // One pointer has the full memory space until the next point is defined.
  // Therefore to read the size of a single shared memory array you need to
  // have a look at the offset for the next array.

  //? input-output tiles
  //* (QtileDim x dimHeads) tiles:
  // qTile (QTileDim x dimHeads)
  scalar_t *qTile = (scalar_t *)sbuf;
  // deltaQTile (QTileDim x dimHeads)
  scalar_t *deltaQTile =
      (scalar_t *)&qTile[QtileDim * (dimHeads + SHARED_MEM_PADDING)];
  // deltaHTile (QTileDim x dimHeads)
  scalar_t *deltaHTile =
      (scalar_t *)&deltaQTile[QtileDim * (dimHeads + SHARED_MEM_PADDING)];

  //* (QtileDim x 1) chunks:
  // nChunk (QTileDim x 1)
  scalar_t *nChunk =
      (scalar_t *)&deltaHTile[QtileDim * (dimHeads + SHARED_MEM_PADDING)];
  // mChunk (QTileDim x 1)
  scalar_t *mChunk = (scalar_t *)&nChunk[QtileDim * (1 + SHARED_MEM_PADDING)];

  //* (KVtileDim x dimHeads) tiles:
  // kTile (KVtileDim x dimHeads)
  scalar_t *kTile = (scalar_t *)&mChunk[QtileDim * (1 + SHARED_MEM_PADDING)];
  // vTile (KVtileDim x dimHeads)
  scalar_t *vTile =
      (scalar_t *)&kTile[KVtileDim * (dimHeads + SHARED_MEM_PADDING)];
  // deltaKTile (KVtileDim x dimHeads)
  scalar_t *deltaKTile =
      (scalar_t *)&vTile[KVtileDim * (dimHeads + SHARED_MEM_PADDING)];
  // deltaVTile (KVtileDim x dimHeads)
  scalar_t *deltaVTile =
      (scalar_t *)&deltaKTile[KVtileDim * (dimHeads + SHARED_MEM_PADDING)];

  //* (KVtileDim x 1) chunks:
  // iChunk (KVtileDim x 1)
  scalar_t *iChunk =
      (scalar_t *)&deltaVTile[KVtileDim * (dimHeads + SHARED_MEM_PADDING)];
  // fChunk (KVtileDim x 1)
  scalar_t *fChunk = (scalar_t *)&iChunk[KVtileDim * (1 + SHARED_MEM_PADDING)];
  // deltaIChunk (KVtileDim x 1)
  scalar_t *deltaIChunk =
      (scalar_t *)&fChunk[KVtileDim * (1 + SHARED_MEM_PADDING)];
  // deltaFChunk (KVtileDim x 1)
  scalar_t *deltaFChunk =
      (scalar_t *)&deltaIChunk[KVtileDim * (1 + SHARED_MEM_PADDING)];

  //? intermediate tiles
  //* (QtileDim x KVtileDim) tiles:
  // sTile (QtileDim x KVtileDim)
  scalar_t *sTile =
      (scalar_t *)&deltaFChunk[KVtileDim * (1 + SHARED_MEM_PADDING)];
  // dddTile (QtileDim x KVtileDim)
  scalar_t *dddTile =
      (scalar_t *)&sTile[QtileDim * (KVtileDim + SHARED_MEM_PADDING)];
  // dcprTile (QtileDim x KVtileDim)
  scalar_t *dcprTile =
      (scalar_t *)&dddTile[QtileDim * (KVtileDim + SHARED_MEM_PADDING)];

  //* (KVtileDim x 1) chunks:
  float *fRowChunk =
      (float *)&dcprTile[QtileDim * (KVtileDim + SHARED_MEM_PADDING)];

  //? flatten the threads to 1D
  const uint flatThreadIdx = blockDim.x * threadIdx.y + threadIdx.x;

  //! PARALLELIZE ALONG BATCHSIZE * NUMHEADS (gridDim.x)
  const uint batchHeadStepQKVdH = seqLen * dimHeads;
  const uint batchHeadStepIFNMgate = seqLen * 1;
  const uint batchHeadStepCD = seqLen * seqLen;
  const uint numBatchHeads = batchSize * numHeads;
  // End for looplevel 0:
  const uint batchHeadEnd = CEIL_DIV(numBatchHeads, gridDim.x);
  // looplevel 0: loop over batches and heads
  for (uint batchHeadIdx = 0; batchHeadIdx < batchHeadEnd; ++batchHeadIdx) {

    // dQ, dK, dV also have this index
    uint batchHeadGridXGlobalMemIdxQKVdH =
        (batchHeadStepQKVdH * gridDim.x) * batchHeadIdx +
        (batchHeadStepQKVdH)*blockIdx.x;

    uint batchHeadGridXGlobalMemIdxIFNMgate =
        (batchHeadStepIFNMgate * gridDim.x) * batchHeadIdx +
        (batchHeadStepIFNMgate)*blockIdx.x;

    uint batchHeadGridXGlobalMemIdxCD =
        (batchHeadStepCD * gridDim.x) * batchHeadIdx +
        (batchHeadStepCD)*blockIdx.x;

#ifdef DEBUG5
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
      printf("B<%d,%d> batchHeadIdx: %d, batchHeadEnd: %d, "
             "batchHeadGridXGlobalMemIdxQKV: "
             "%d\n",
             blockIdx.x, blockIdx.y, batchHeadIdx, batchHeadEnd,
             batchHeadGridXGlobalMemIdxQKV);
    }
#endif

    //! PARALLELIZE ALONG SEQLEN (gridDim.y)
    // looplevel 1 (j-loop): loop over KVtile blocks along seqLen dim
    // Ends for looplevel 1:
    const uint kvTileEnd = CEIL_DIV(seqLen, KVtileDim * gridDim.y);
    for (uint kvTileIdx = 0; kvTileIdx < kvTileEnd; ++kvTileIdx) {

      //* kTile, vTile Global Memory Index
      // (grid&block) offset in K,V matrix for kTile&vTile (global memory)
      const uint kvTileGridXYGlobalMemIdx =
          batchHeadGridXGlobalMemIdxQKVdH +
          (dimHeads * KVtileDim * gridDim.y) * kvTileIdx;
      const uint kvTileBlockGlobalMemIdx =
          kvTileGridXYGlobalMemIdx + (dimHeads * KVtileDim) * blockIdx.y;

      //* sTile Global Memory Index (virtual, as never materialized fully)
      // (grid&block) offset Y-axis in S = Q*K^T matrix (along sequence
      // dimension) (used for checking causality)
      // Note: we add "Xdim" to the name to indicate that this is the
      // corresponding x/column-dimension in the S matrix (differs from the
      // gridDim.y)
      const uint sTileXdimGridYIdx = KVtileDim * gridDim.y * kvTileIdx;
      const uint sTileXdimBlockYIdx =
          sTileXdimGridYIdx + KVtileDim * blockIdx.y;

      const uint ifChunkGridXYGlobalMemIdx =
          batchHeadGridXGlobalMemIdxIFNMgate +
          (1 * KVtileDim * gridDim.y) * kvTileIdx;
      const uint ifChunkBlockGlobalMemIdx =
          ifChunkGridXYGlobalMemIdx + (1 * KVtileDim) * blockIdx.y;

      //! Load iChunk & fChunk, Init deltaIChunk & deltaFChunk to zero in SRAM
      const uint ifChunkEnd = CEIL_DIV(KVtileDim, blockDim.x * blockDim.y);
      for (uint ifChunkIdx = 0; ifChunkIdx < ifChunkEnd; ++ifChunkIdx) {
        //? if idxes
        //* shared memory
        const uint ifThreadSharedMemIdx =
            flatThreadIdx + blockDim.x * blockDim.y * ifChunkIdx;
        //* global memory
        const uint ifThreadGlobalMemIdx =
            ifChunkBlockGlobalMemIdx + flatThreadIdx;

        if (ifThreadSharedMemIdx < KVtileDim) {
          SMEMVECTOR(iChunk, ifThreadSharedMemIdx) =
              iGatePreact[ifThreadGlobalMemIdx];
          SMEMVECTOR(fChunk, ifThreadSharedMemIdx) =
              logsigmoid_g(fGatePreact[ifThreadGlobalMemIdx]);
          // without logsigmoid for debugging only:
          //   SMEMVECTOR(fChunk, ifThreadSharedMemIdx) =
          //       fGatePreact[ifThreadGlobalMemIdx];
          SMEMVECTOR(deltaIChunk, ifThreadSharedMemIdx) =
              dscalar_zero<scalar_t>();
          SMEMVECTOR(deltaFChunk, ifThreadSharedMemIdx) =
              dscalar_zero<scalar_t>();
        }
      }
      __syncthreads();

      //! Load kTile & vTile, Init deltaKTile & deltaVTile to zero in SRAM
      // loops over rows (outer) and columns (inner) of kTile and vTile
      const uint kvWarpTileYEnd = CEIL_DIV(KVtileDim, blockDim.y); // rows
      const uint kvWarpTileXEnd = CEIL_DIV(dimHeads, blockDim.x);  // cols
      for (uint kvWarpTileYIdx = 0; kvWarpTileYIdx < kvWarpTileYEnd;
           ++kvWarpTileYIdx) {
#ifdef DEBUG2
        if ((blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0) &&
            (threadIdx.y == 0)) {
          printf("kvWarpTileYIdx=%d: kvWarpTileYEnd: %d, kvWarpTileXEnd: %d\n",
                 kvWarpTileYIdx, kvWarpTileYEnd, kvWarpTileXEnd);
        }
#endif
        for (uint kvWarpTileXIdx = 0; kvWarpTileXIdx < kvWarpTileXEnd;
             ++kvWarpTileXIdx) {
          //? kvWarpTileIdxes for k-tile AND v-tile
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
          // just kept for reference:
          // //! while loading k: k = k / sqrt(dimHeads)
          // SMEMARRAY(kTile, dimHeads, kvWarpTileThreadSharedMemYIdx,
          //           kvWarpTileThreadSharedMemXIdx) =
          //     mul_g(matK[kvWarpTileThreadGlobalMemIdx],
          //           float2type<scalar_t>(rsqrtf(type2float((dimHeads)))));
          //! For simplicity: assume that KVtileDim is a multiple of blockDim.y
          //! and dimHeads is a multiple of blockDim.x
          SMEMARRAY(kTile, dimHeads, kvWarpTileThreadSharedMemYIdx,
                    kvWarpTileThreadSharedMemXIdx) =
              matK[kvWarpTileThreadGlobalMemIdx];
          SMEMARRAY(vTile, dimHeads, kvWarpTileThreadSharedMemYIdx,
                    kvWarpTileThreadSharedMemXIdx) =
              matV[kvWarpTileThreadGlobalMemIdx];
        }
      }
      __syncthreads();

      // looplevel 2 (i-loop): loop over QTile blocks along seqLen dim
      const uint qTileEnd = CEIL_DIV(seqLen, QtileDim);
      uint jIdx = blockIdx.y + kvTileIdx * gridDim.y;
      uint qTileStart = FLOOR_DIV(jIdx * KVtileDim, QtileDim);
      for (uint qTileIdx = qTileStart; qTileIdx < qTileEnd; ++qTileIdx) {

        //* qTile Global Memory Index
        const uint qdHTileBlockGlobalMemIdx =
            batchHeadGridXGlobalMemIdxQKVdH + (dimHeads * QtileDim) * qTileIdx;

        //* nChunk, mChunk Global Memory Index
        const uint nmChunkBlockGlobalMemIdx =
            batchHeadGridXGlobalMemIdxIFNMgate + (1 * QtileDim) * qTileIdx;

        //* sTile Global Memory Index
        const uint sTileYdimGridYIdx = QtileDim * qTileIdx;
        const uint sTileYdimBlockYIdx = sTileYdimGridYIdx;

        //! Load nChunk & mChunk in SRAM
        const uint nmChunkEnd = CEIL_DIV(QtileDim, blockDim.x * blockDim.y);
        for (uint nmChunkIdx = 0; nmChunkIdx < nmChunkEnd; ++nmChunkIdx) {
          //? nm idxes
          //* shared memory
          const uint nmThreadSharedMemIdx =
              flatThreadIdx + blockDim.x * blockDim.y * nmChunkIdx;
          //* global memory
          const uint nmThreadGlobalMemIdx =
              nmChunkBlockGlobalMemIdx + flatThreadIdx;

          if (nmThreadSharedMemIdx < QtileDim) {
            SMEMVECTOR(nChunk, nmThreadSharedMemIdx) =
                vecN[nmThreadGlobalMemIdx];
            SMEMVECTOR(mChunk, nmThreadSharedMemIdx) =
                vecM[nmThreadGlobalMemIdx];
          }
        }

        //! Load qTile & deltaHTile in SRAM
        // loops over rows (outer) and columns (inner) of qTile and deltaHTile
        const uint qdHWarpTileYEnd = CEIL_DIV(QtileDim, blockDim.y); // rows
        const uint qdHWarpTileXEnd = CEIL_DIV(dimHeads, blockDim.x); // cols
        for (uint qdHWarpTileYIdx = 0; qdHWarpTileYIdx < qdHWarpTileYEnd;
             ++qdHWarpTileYIdx) {
          for (uint qdHWarpTileXIdx = 0; qdHWarpTileXIdx < qdHWarpTileXEnd;
               ++qdHWarpTileXIdx) {
            //? qdHWarpTileIdxes for q-tile AND delta-h-tile
            //* shared memory:
            const uint qdHWarpTileThreadSharedMemYIdx =
                blockDim.y * qdHWarpTileYIdx + threadIdx.y;
            const uint qdHWarpTileThreadSharedMemXIdx =
                blockDim.x * qdHWarpTileXIdx + threadIdx.x;
            //* global memory:
            // left upper corner of qTileBlock in Q (global memory)
            const uint qdHWarpTileBlockGlobalMemIdx =
                qdHTileBlockGlobalMemIdx +
                (dimHeads * blockDim.y) * qdHWarpTileYIdx +
                blockDim.x * qdHWarpTileXIdx;
            const uint qdHWarpTileThreadGlobalMemIdx =
                qdHWarpTileBlockGlobalMemIdx + dimHeads * threadIdx.y +
                threadIdx.x;
            SMEMARRAY(qTile, dimHeads, qdHWarpTileThreadSharedMemYIdx,
                      qdHWarpTileThreadSharedMemXIdx) =
                matQ[qdHWarpTileThreadGlobalMemIdx];
            SMEMARRAY(deltaHTile, dimHeads, qdHWarpTileThreadSharedMemYIdx,
                      qdHWarpTileThreadSharedMemXIdx) =
                deltaH[qdHWarpTileThreadGlobalMemIdx];
          }
        }

        //! Compute deltaCTile = deltaHtile  vTile^T (and divide by nChunk)
        // TODO

        //! Compute sTile = (qTile  kTile^T) * (1/sqrt(d))
        // TODO

        //! Compute dDtile = deltaCTile * sTile
        // TODO

        //! sum up deltaIChunk & deltaFChunk and update in SRAM
        // TODO

        //! Construct D'Tile from fChunk and iChunk
        // TODO

        //! Compute pTile = deltaCTile * D'Tile
        // TODO

        //! Compute deltaQTile = pTile  (kTile/sqrt(d))
        // TODO

        //! Atomic add deltaQTile to deltaQ in HBM (HOW TO DO??)
        // TODO check how to do this

        //! Compute deltaKTile = pTile^T  (qTile/sqrt(d)) and update in SRAM
        // TODO

        //! Compute rTile = sTile * D'Tile
        // TODO

        //! Compute deltaVTile = rTile^T  deltaHTile and update in SRAM
        // TODO
      } // end looplevel 2 (i-loop)
        //! Store deltaKTile & deltaVTile in HBM
        // TODO
        //! Store deltaIChunk & deltaFChunk in HBM
        // TODO
    }   // end looplevel 1 (j-loop)
  }     // end looplevel 0
} // kernels::vlstm_fw

template <typename scalar_t>
void kernel_dispatchers::vlstm_bw_dispatch(
    scalar_t *deltaQ, scalar_t *deltaK, scalar_t *deltaV,
    scalar_t *deltaIGatePreact, scalar_t *deltaFGatePreact, scalar_t *deltaH,
    scalar_t *matQ, scalar_t *matK, scalar_t *matV, scalar_t *iGatePreact,
    scalar_t *fGatePreact, scalar_t *vecN, scalar_t *vecM, int batchSize,
    int numHeads, int seqLen, int dimHeads) {
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
  //   const dim3 gridDims(1, 1);

  //! calculate dynamic shared memory size
  // TODO understand how memory padding works!
  // Why at innermost dim? Because memory is organized consecutively

  //? input-output tiles
  const uint qdQdHTileSharedMemSize =
      sizeof(scalar_t) * QtileDim * (dimHeads + SHARED_MEM_PADDING);
  const uint kvdKdVTileSharedMemSize =
      sizeof(scalar_t) * KVtileDim * (dimHeads + SHARED_MEM_PADDING);

  // See here:
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses
  // the idea of the padding is that every number is stored in a different
  // memory bank this should help to avoid bank conflicts as many threads need
  // to access the same input and forget gate values at the same time for the
  // gate matrix computation
  // TODO check if this is really helping!
  const uint ifdidfChunkSharedMemSize =
      sizeof(scalar_t) * KVtileDim * (1 + SHARED_MEM_PADDING);

  const uint nmChunkSharedMemSize =
      sizeof(scalar_t) * QtileDim * (1 + SHARED_MEM_PADDING);

  //? intermediate tiles
  const uint sdprdcddTileSharedMemSize =
      sizeof(scalar_t) * QtileDim * (KVtileDim + SHARED_MEM_PADDING);

  // we keep these as float as it acts as accumulator
  const uint fTileRowSharedMemSize =
      sizeof(float) * KVtileDim * (1 + SHARED_MEM_PADDING);

  // Input/Output tiles: 4x for qTile, vTile, kTile, hTile
  // Intermediate tiles: 2x for cTile, dTile
  // Intermediate tiles: 2x for mChunk, lChunk
  const uint sharedMemorySize =
      3 * qdQdHTileSharedMemSize + 4 * kvdKdVTileSharedMemSize +
      4 * ifdidfChunkSharedMemSize + 2 * nmChunkSharedMemSize +
      3 * sdprdcddTileSharedMemSize + 1 * fTileRowSharedMemSize;

  printf("blocksxy: %d-%d, threadsxy: %d-%d, shared_mem in bytes: %d\n",
         gridDims.x, gridDims.y, blockDims.x, blockDims.y, sharedMemorySize);
  // cudaSetDevice(0);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  auto kernel = kernels::vlstm_bw<scalar_t, TblockDim, QtileDim, KVtileDim>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       sharedMemorySize);
  // define void* pointers to the kernel arguments
  // TODO adapt this!
  void *kernelArgs[] = {(void *)&deltaQ,
                        (void *)&deltaK,
                        (void *)&deltaV,
                        (void *)&deltaIGatePreact,
                        (void *)&deltaFGatePreact,
                        (void *)&deltaH,
                        (void *)&matQ,
                        (void *)&matK,
                        (void *)&matV,
                        (void *)&iGatePreact,
                        (void *)&fGatePreact,
                        (void *)&vecN,
                        (void *)&vecM,
                        (void *)&batchSize,
                        (void *)&numHeads,
                        (void *)&seqLen,
                        (void *)&dimHeads};

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
template void kernel_dispatchers::vlstm_bw_dispatch<__nv_bfloat16>(
    __nv_bfloat16 *deltaQ, __nv_bfloat16 *deltaK, __nv_bfloat16 *deltaV,
    __nv_bfloat16 *deltaIGatePreact, __nv_bfloat16 *deltaFGatePreact,
    __nv_bfloat16 *deltaH, __nv_bfloat16 *matQ, __nv_bfloat16 *matK,
    __nv_bfloat16 *matV, __nv_bfloat16 *iGatePreact, __nv_bfloat16 *fGatePreact,
    __nv_bfloat16 *vecN, __nv_bfloat16 *vecM, int batchSize, int numHeads,
    int seqLen, int dimHeads);
template void kernel_dispatchers::vlstm_bw_dispatch<__half>(
    __half *deltaQ, __half *deltaK, __half *deltaV, __half *deltaIGatePreact,
    __half *deltaFGatePreact, __half *deltaH, __half *matQ, __half *matK,
    __half *matV, __half *iGatePreact, __half *fGatePreact, __half *vecN,
    __half *vecM, int batchSize, int numHeads, int seqLen, int dimHeads);
template void kernel_dispatchers::vlstm_bw_dispatch<float>(
    float *deltaQ, float *deltaK, float *deltaV, float *deltaIGatePreact,
    float *deltaFGatePreact, float *deltaH, float *matQ, float *matK,
    float *matV, float *iGatePreact, float *fGatePreact, float *vecN,
    float *vecM, int batchSize, int numHeads, int seqLen, int dimHeads);
} // namespace vlstm