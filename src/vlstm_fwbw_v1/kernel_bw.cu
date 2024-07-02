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
                         scalar_t *matC, scalar_t *deltaH, scalar_t *matQ,
                         scalar_t *matK, scalar_t *matV, scalar_t *iGatePreact,
                         scalar_t *fGatePreact, scalar_t *vecN, scalar_t *vecM,
                         float *csDeltaDTildeChunkArr, float *csDeltaDTildeVec,
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
// #define OUTPUTdDTile 1
// #define OUTPUTdDtildeTile 1
// #define OUTPUTDTile 1
// #define OUTPUTDcsTile 1
#define OUTPUTPRTile 1
#define OUTPUTPRTileR 1

// #define DEBUG_WRdeltaI 1
// #define DEBUG_deltaISUM0 1
// #define DEBUG_deltaISUM1 1
// #define DEBUG_deltaISUM2 1

// #define DEBUG_IJIDX 1

// #define DEBUG_DeltaDCS0 1
// #define DEBUG_DeltaDCS1 1
// #define DEBUG_DeltaDCS2 1

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
                  scalar_t *matC, scalar_t *deltaH, scalar_t *matQ,
                  scalar_t *matK, scalar_t *matV, scalar_t *iGatePreact,
                  scalar_t *fGatePreact, scalar_t *vecN, scalar_t *vecM,
                  float *csDeltaDTildeChunkArr, float *csDeltaDTildeVec,
                  int batchSize, int numHeads, int seqLen, int dimHeads) {
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
  // TODO can we define multiple gridGroups?
  // -> e.g. all threadblocks with the same gridDim.x should go into one group
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
  // mDeltaDcsCorrChunk (QTileDim x 1)
  scalar_t *mDeltaDcsCorrChunk =
      (scalar_t *)&nChunk[QtileDim * (1 + SHARED_MEM_PADDING)];

  //* (KVtileDim x dimHeads) tiles:
  // kTile (KVtileDim x dimHeads)
  scalar_t *kTile =
      (scalar_t *)&mDeltaDcsCorrChunk[QtileDim * (1 + SHARED_MEM_PADDING)];
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
      (scalar_t *)&fChunk[QtileDim * (1 + SHARED_MEM_PADDING)];
  // deltaFChunk (KVtileDim x 1)
  scalar_t *deltaFChunk =
      (scalar_t *)&deltaIChunk[KVtileDim * (1 + SHARED_MEM_PADDING)];

  //? intermediate tiles
  //* (QtileDim x KVtileDim) tiles:
  // dstrRTile (QtileDim x KVtileDim)
  scalar_t *dstrRTile =
      (scalar_t *)&deltaFChunk[KVtileDim * (1 + SHARED_MEM_PADDING)];
  // sTile (QtileDim x KVtileDim)
  scalar_t *sPTile =
      (scalar_t *)&dstrRTile[KVtileDim * (1 + SHARED_MEM_PADDING)];
  // dDTile (QtileDim x KVtileDim)
  scalar_t *dDTile =
      (scalar_t *)&sPTile[QtileDim * (KVtileDim + SHARED_MEM_PADDING)];
  // dCDcsTile (QtileDim x KVtileDim)
  scalar_t *dCDcsTile =
      (scalar_t *)&dDTile[QtileDim * (KVtileDim + SHARED_MEM_PADDING)];

  //* (KVtileDim x 1) chunks:
  float *fAccRowChunk =
      (float *)(&(dCDcsTile[QtileDim * (KVtileDim + SHARED_MEM_PADDING)]));

  //? flatten the threads to 1D
  const uint flatThreadIdx = blockDim.x * threadIdx.y + threadIdx.x;

  //! PARALLELIZE ALONG BATCHSIZE * NUMHEADS (gridDim.x)
  const uint batchHeadStepQKVdH = seqLen * dimHeads;
  const uint batchHeadStepIFNMgate =
      seqLen * 1; // TODO rename: csDeltaDTildeVec has same step
  const uint batchHeadStepCD = seqLen * seqLen;
  const uint batchHeadStepDeltaDcsChunkArr = gridDim.y * QtileDim;
  const uint numBatchHeads = batchSize * numHeads;
  // End for looplevel 0:
  const uint batchHeadEnd = CEIL_DIV(numBatchHeads, gridDim.x);
  //! looplevel 0: loop over batches and heads
  for (uint batchHeadIdx = 0; batchHeadIdx < batchHeadEnd; ++batchHeadIdx) {

    // dQ, dK, dV also have this index
    const uint batchHeadGridXGlobalMemIdxQKVdH =
        (batchHeadStepQKVdH * gridDim.x) * batchHeadIdx +
        (batchHeadStepQKVdH)*blockIdx.x;

    const uint batchHeadGridXGlobalMemIdxIFNMgate =
        (batchHeadStepIFNMgate * gridDim.x) * batchHeadIdx +
        (batchHeadStepIFNMgate)*blockIdx.x;

    const uint batchHeadGridXGlobalMemIdxCD =
        (batchHeadStepCD * gridDim.x) * batchHeadIdx +
        (batchHeadStepCD)*blockIdx.x;

    const uint batchHeadGridXGlobalMemIdxDeltaDcsChunkArr =
        (batchHeadStepDeltaDcsChunkArr * gridDim.x) * batchHeadIdx +
        (batchHeadStepDeltaDcsChunkArr)*blockIdx.x;

#ifdef DEBUG5
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
      printf("B<%d,%d> batchHeadIdx: %d, batchHeadEnd: %d, "
             "batchHeadGridXGlobalMemIdxQKV: "
             "%d\n",
             blockIdx.x, blockIdx.y, batchHeadIdx, batchHeadEnd,
             batchHeadGridXGlobalMemIdxQKV);
    }
#endif

    //! Initialize deltaQ, deltaK, deltaV to zero in HBM
    // already done in the kernel dispatcher, via torch::zeros()

    //! Initialize csDeltaDTildeVec to zero in HBM
    // already done in the kernel dispatcher, via cudaMemset

    //! PARALLELIZE ALONG SEQLEN (gridDim.y)
    //! looplevel 1 (j-loop): loop over KVtile blocks along seqLen dim
    // Ends for looplevel 1:
    const uint kvTileEnd = CEIL_DIV(seqLen, KVtileDim * gridDim.y);
    for (uint kvTileIdx = 0; kvTileIdx < kvTileEnd; ++kvTileIdx) {

      //* kTile, vTile Global Memory Index
      // (grid&block) offset in K,V matrix for kTile&vTile (global memory)
      const uint kvdKdVTileGridXYGlobalMemIdx =
          batchHeadGridXGlobalMemIdxQKVdH +
          (dimHeads * KVtileDim * gridDim.y) * kvTileIdx;
      const uint kvdKdVTileBlockGlobalMemIdx =
          kvdKdVTileGridXYGlobalMemIdx + (dimHeads * KVtileDim) * blockIdx.y;

      //* sTile Global Memory Index (virtual, as never materialized fully)
      // (grid&block) offset Y-axis in S = Q*K^T matrix (along sequence
      // dimension) (used for checking causality)
      // Note: we add "Xdim" to the name to indicate that this is the
      // corresponding x/column-dimension in the S matrix (differs from the
      // gridDim.y)
      const uint sTileXdimGridYIdx = KVtileDim * gridDim.y * kvTileIdx;
      const uint sTileXdimBlockYIdx =
          sTileXdimGridYIdx + KVtileDim * blockIdx.y;

      const uint iChunkGridXYGlobalMemIdx =
          batchHeadGridXGlobalMemIdxIFNMgate +
          (1 * KVtileDim * gridDim.y) * kvTileIdx;
      const uint iChunkBlockGlobalMemIdx =
          iChunkGridXYGlobalMemIdx + (1 * KVtileDim) * blockIdx.y;

      //! Load iChunk, Init deltaIChunk, deltaFChunk & fAccRowChunk to
      //! zero in SRAM
      const uint idFdIChunkEnd = CEIL_DIV(KVtileDim, blockDim.x * blockDim.y);
      for (uint idFdIChunkIdx = 0; idFdIChunkIdx < idFdIChunkEnd;
           ++idFdIChunkIdx) {
        //? idFdI idxes
        //* shared memory
        const uint idFdIThreadSharedMemIdx =
            flatThreadIdx + blockDim.x * blockDim.y * idFdIChunkIdx;
        //* global memory
        const uint iThreadGlobalMemIdx =
            iChunkBlockGlobalMemIdx + flatThreadIdx;

        if (idFdIThreadSharedMemIdx < KVtileDim) {
          SMEMVECTOR(iChunk, idFdIThreadSharedMemIdx) =
              iGatePreact[iThreadGlobalMemIdx];
          SMEMVECTOR(deltaIChunk, idFdIThreadSharedMemIdx) =
              dscalar_zero<scalar_t>();
          SMEMVECTOR(deltaFChunk, idFdIThreadSharedMemIdx) =
              dscalar_zero<scalar_t>();
          SMEMVECTOR(fAccRowChunk, idFdIThreadSharedMemIdx) = 0.0f;
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
              kvdKdVTileBlockGlobalMemIdx +
              (dimHeads * blockDim.y) * kvWarpTileYIdx +
              blockDim.x * kvWarpTileXIdx;
          const uint kvWarpTileThreadGlobalMemIdx =
              kvWarpTileBlockGlobalMemIdx + dimHeads * threadIdx.y +
              threadIdx.x;

          //! For simplicity: assume that KVtileDim is a multiple of blockDim.y
          //! and dimHeads is a multiple of blockDim.x
          SMEMARRAY(kTile, dimHeads, kvWarpTileThreadSharedMemYIdx,
                    kvWarpTileThreadSharedMemXIdx) =
              matK[kvWarpTileThreadGlobalMemIdx];
          SMEMARRAY(vTile, dimHeads, kvWarpTileThreadSharedMemYIdx,
                    kvWarpTileThreadSharedMemXIdx) =
              matV[kvWarpTileThreadGlobalMemIdx];

          // init deltaKTile and deltaVTile to zero
          SMEMARRAY(deltaKTile, dimHeads, kvWarpTileThreadSharedMemYIdx,
                    kvWarpTileThreadSharedMemXIdx) = dscalar_zero<scalar_t>();
          SMEMARRAY(deltaVTile, dimHeads, kvWarpTileThreadSharedMemYIdx,
                    kvWarpTileThreadSharedMemXIdx) = dscalar_zero<scalar_t>();
        }
      }
      __syncthreads();

      //! looplevel 2 (i-loop): loop over QTile blocks along seqLen dim
      const uint qTileEnd = CEIL_DIV(seqLen, QtileDim);
      const uint jIdx = blockIdx.y + kvTileIdx * gridDim.y;
      //   const uint qTileStart = FLOOR_DIV(jIdx * KVtileDim, QtileDim);
      //? We start from the first QTile such that we can sync the threadblocks
      //? globally, and ensure that they are in sync for the next KVtile
      //? they need to be in sync for cumsum(deltaDtildeTile) and qTile
      const uint qTileStart = 0;
      for (uint qTileIdx = qTileStart; qTileIdx < qTileEnd; ++qTileIdx) {

        const uint iIdx = qTileIdx;
        //* qTile Global Memory Index
        const uint qdHdQTileBlockGlobalMemIdx =
            batchHeadGridXGlobalMemIdxQKVdH + (dimHeads * QtileDim) * qTileIdx;

        //* nChunk, mDeltaDcsCorrChunk, fChunk Global Memory Index
        const uint nmfChunkBlockGlobalMemIdx =
            batchHeadGridXGlobalMemIdxIFNMgate + (1 * QtileDim) * qTileIdx;

        //* sTile Global Memory Index
        const uint sTileYdimGridYIdx = QtileDim * qTileIdx;
        const uint sTileYdimBlockYIdx = sTileYdimGridYIdx;

#ifdef DEBUG_IJIDX
        if ((blockIdx.x == 0) && (blockIdx.y <= 1) && (flatThreadIdx == 0)) {
          printf("blockIdx(x,y)=(%d,%d), ijIdx(i,j)=(%d,%d), "
                 "sTileXYIdx(x,y)=(%d,%d), kvTileIdx=%d, qTileIdx=%d\n",
                 blockIdx.x, blockIdx.y, iIdx, jIdx, sTileXdimBlockYIdx,
                 sTileYdimBlockYIdx, kvTileIdx, qTileIdx);
        }
#endif

        //! Load nChunk, mDeltaDcsCorrChunk, fChunk in SRAM
        const uint nmfChunkEnd = CEIL_DIV(QtileDim, blockDim.x * blockDim.y);
        for (uint nmfChunkIdx = 0; nmfChunkIdx < nmfChunkEnd; ++nmfChunkIdx) {
          //? nmf idxes
          //* shared memory
          const uint nmfThreadSharedMemIdx =
              flatThreadIdx + blockDim.x * blockDim.y * nmfChunkIdx;
          //* global memory
          const uint nmfThreadGlobalMemIdx =
              nmfChunkBlockGlobalMemIdx + flatThreadIdx;

          if (nmfThreadSharedMemIdx < QtileDim) {
            SMEMVECTOR(nChunk, nmfThreadSharedMemIdx) =
                vecN[nmfThreadGlobalMemIdx];
            SMEMVECTOR(mDeltaDcsCorrChunk, nmfThreadSharedMemIdx) =
                vecM[nmfThreadGlobalMemIdx];

            // load raw preactivations for D matrix and deltaF
            SMEMVECTOR(fChunk, nmfThreadSharedMemIdx) =
                fGatePreact[nmfThreadGlobalMemIdx];
          }
        }

        //! Load qTile & deltaHTile in SRAM
        // (QTileDim x dimHeads)
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
                qdHdQTileBlockGlobalMemIdx +
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
        __syncthreads();
        //? Global Sync
        gridGroup.sync();

        //! Compute deltaCTile = (deltaHtile  vTile^T) / nChunk (and divide by
        //! nChunk)
        // (QtileDim,KVtileDim) = (QtileDim,dimHeads) x (dimHeads,KVtileDim)
        // loops over cTile rows (outer) and columns (inner)
        const uint dCWarpTileYEnd = CEIL_DIV(QtileDim, blockDim.y);
        const uint dCWarpTileXEnd = CEIL_DIV(KVtileDim, blockDim.x);
        for (uint dCWarpTileYIdx = 0; dCWarpTileYIdx < dCWarpTileYEnd;
             ++dCWarpTileYIdx) {

          for (uint dCWarpTileXIdx = 0; dCWarpTileXIdx < dCWarpTileXEnd;
               ++dCWarpTileXIdx) {
            //? cTileIdxes
            //* shared memory:
            const uint cWarpTileThreadSharedMemYIdx =
                blockDim.y * dCWarpTileYIdx + threadIdx.y;
            const uint cWarpTileThreadSharedMemXIdx =
                blockDim.x * dCWarpTileXIdx + threadIdx.x;

            // scalar_t qk_acc = dscalar_zero<scalar_t>();
            float acc = 0.0f;
            for (uint i = 0; i < dimHeads; ++i) {
              acc =
                  add_g(acc, type2float(mul_g(
                                 SMEMARRAY(deltaHTile, dimHeads,
                                           cWarpTileThreadSharedMemYIdx, i),
                                 SMEMARRAY(vTile, dimHeads,
                                           cWarpTileThreadSharedMemXIdx, i))));
            }
            // dC = deltaH *V^T / n
            scalar_t nChunkVal =
                SMEMVECTOR(nChunk, cWarpTileThreadSharedMemYIdx);

            // we first cast to scalar_t and then divide (this will also the
            // case for tensor cores)
            SMEMARRAY(dCDcsTile, KVtileDim, cWarpTileThreadSharedMemYIdx,
                      cWarpTileThreadSharedMemXIdx) =
                div_g(float2type<scalar_t>(acc), nChunkVal);
          }
        }
        __syncthreads();

        //! Compute sTile = (qTile  kTile^T) * (1/sqrt(d)) and
        //! dDTile = deltaCTile * sTile (pointwise)
        // (QtileDim,KVtileDim) = (QtileDim,dimHeads) x (dimHeads,KVtileDim)
        // loops over sTile rows (outer) and columns (inner)
        const uint sWarpTileYEnd = CEIL_DIV(QtileDim, blockDim.y);
        const uint sWarpTileXEnd = CEIL_DIV(KVtileDim, blockDim.x);
        for (uint sWarpTileYIdx = 0; sWarpTileYIdx < sWarpTileYEnd;
             ++sWarpTileYIdx) {

          for (uint sWarpTileXIdx = 0; sWarpTileXIdx < sWarpTileXEnd;
               ++sWarpTileXIdx) {
            //? cTileIdxes
            //* shared memory:
            const uint sWarpTileThreadSharedMemYIdx =
                blockDim.y * sWarpTileYIdx + threadIdx.y;
            const uint sWarpTileThreadSharedMemXIdx =
                blockDim.x * sWarpTileXIdx + threadIdx.x;

            // scalar_t qk_acc = dscalar_zero<scalar_t>();
            float acc = 0.0f;
            for (uint i = 0; i < dimHeads; ++i) {
              acc =
                  add_g(acc, type2float(mul_g(
                                 SMEMARRAY(qTile, dimHeads,
                                           sWarpTileThreadSharedMemYIdx, i),
                                 SMEMARRAY(kTile, dimHeads,
                                           sWarpTileThreadSharedMemXIdx, i))));
            }
            // compute sTile
            scalar_t s_val =
                float2type<scalar_t>(mul_g(acc, rsqrtf(type2float(dimHeads))));
            SMEMARRAY(sPTile, KVtileDim, sWarpTileThreadSharedMemYIdx,
                      sWarpTileThreadSharedMemXIdx) = s_val;
            // compute dDTile
            scalar_t deltaC_val =
                SMEMARRAY(dCDcsTile, KVtileDim, sWarpTileThreadSharedMemYIdx,
                          sWarpTileThreadSharedMemXIdx);
            scalar_t ddd_val = mul_g(deltaC_val, s_val);
            SMEMARRAY(dDTile, KVtileDim, sWarpTileThreadSharedMemYIdx,
                      sWarpTileThreadSharedMemXIdx) = ddd_val;
          }
        }
        __syncthreads();

        //! Compute dDtile = deltaCTile * sTile
        // Done with the pointwise multiplication in the previous step

#ifdef OUTPUTdDTile
        //! DEBUG: write dDtile to global memory
        // left upper corner of cWarpTileBlock in C (global memory)
        //* cdTile Global Memory Index (Debug only)
        const uint cdTileGridXYGlobalMemIdx =
            batchHeadGridXGlobalMemIdxCD + (seqLen * QtileDim) * qTileIdx;
        const uint cdTileBlockGlobalMemIdx =
            cdTileGridXYGlobalMemIdx + (kvTileIdx * KVtileDim * gridDim.y) +
            (1 * KVtileDim) * blockIdx.y;

        const uint cdWarpTileYEnd = CEIL_DIV(QtileDim, blockDim.y);
        const uint cdWarpTileXEnd = CEIL_DIV(KVtileDim, blockDim.x);
        for (uint cdWarpTileYIdx = 0; cdWarpTileYIdx < cdWarpTileYEnd;
             ++cdWarpTileYIdx) {
          for (uint cdWarpTileXIdx = 0; cdWarpTileXIdx < cdWarpTileXEnd;
               ++cdWarpTileXIdx) {
            //? cTileIdxes
            //* shared memory:
            const uint cdWarpTileThreadSharedMemYIdx =
                blockDim.y * cdWarpTileYIdx + threadIdx.y;
            const uint cdWarpTileThreadSharedMemXIdx =
                blockDim.x * cdWarpTileXIdx + threadIdx.x;
            //* global memory:
            const uint cdWarpTileBlockGlobalMemIdx =
                cdTileBlockGlobalMemIdx +
                (seqLen * blockDim.y) * cdWarpTileYIdx +
                blockDim.x * cdWarpTileXIdx;
            const uint cdWarpTileThreadGlobalMemIdx =
                cdWarpTileBlockGlobalMemIdx + seqLen * threadIdx.y +
                threadIdx.x;

            matC[cdWarpTileThreadGlobalMemIdx] =
                SMEMARRAY(dDTile, KVtileDim, cdWarpTileThreadSharedMemYIdx,
                          cdWarpTileThreadSharedMemXIdx);
          }
        }
#endif
        //! Construct D'Tile from fChunk and iChunk and
        //! compute deltaDtildeTile = deltaDTile * D'Tile
        // flatten all threads to 1D along kvTileDim (j-direction),
        // sum up the f gate values in i-direction (qTileDim),
        // store the last row of the D'Tile in fAccRowChunk
        // take care of causality

        // loop in j-direction (kvTileDim / x-dim)
        const uint dTileXdimEnd = CEIL_DIV(KVtileDim, blockDim.x * blockDim.y);
        for (uint dTileXdimIdx = 0; dTileXdimIdx < dTileXdimEnd;
             ++dTileXdimIdx) {
          //? dTile idxes
          //* shared memory
          const uint dTileXdimThreadSharedMemIdx =
              flatThreadIdx + blockDim.x * blockDim.y * dTileXdimIdx;

          //* dTile global index (virtual, as never materialized fully)
          const uint dTileXdimThreadIdx =
              sTileXdimBlockYIdx + dTileXdimThreadSharedMemIdx;

          if (dTileXdimThreadSharedMemIdx < KVtileDim) {
            // pre load the i_val for the current column
            const scalar_t i_val =
                SMEMVECTOR(iChunk, dTileXdimThreadSharedMemIdx);

            // sum up f gate values in i-direction
            float f_acc = SMEMVECTOR(fAccRowChunk, dTileXdimThreadSharedMemIdx);
            // loop in j-direction (qTileDim / y-dim)
            for (uint dTileYdimThreadSharedMemIdx = 0;
                 dTileYdimThreadSharedMemIdx < QtileDim;
                 ++dTileYdimThreadSharedMemIdx) {

              //* dTile global index (virtual, as never materialized fully)
              const uint dTileYdimThreadIdx =
                  sTileYdimBlockYIdx + dTileYdimThreadSharedMemIdx;

              //? Compute f gate cumsum entries in dTile
              // only sum up the f gates in the lower triangular (take care of
              // causality)
              if (dTileYdimThreadIdx > dTileXdimThreadIdx) {
                scalar_t f_val = logsigmoid_g(
                    SMEMVECTOR(fChunk, dTileYdimThreadSharedMemIdx));
                f_acc = add_g(f_acc, type2float(f_val));
              }
              // store the last row of D'Tile in fAccRowChunk
              if (dTileYdimThreadSharedMemIdx == QtileDim - 1) {
                SMEMVECTOR(fAccRowChunk, dTileXdimThreadSharedMemIdx) = f_acc;
              }

              //? Create D'Tile entries sum(f) + i
              //? Create deltaDtildeTile entries (overwrite the dDTile entries)
              scalar_t d_val = dscalar_zero<scalar_t>();
              scalar_t deltaDtilde_val = dscalar_zero<scalar_t>();
              if (dTileYdimThreadIdx < dTileXdimThreadIdx) {
                // (-> in upper triangular part)
                // d_val = 0;
              } else {
                scalar_t deltaD_val =
                    SMEMARRAY(dDTile, KVtileDim, dTileYdimThreadSharedMemIdx,
                              dTileXdimThreadSharedMemIdx);
                scalar_t m_val =
                    SMEMVECTOR(mDeltaDcsCorrChunk, dTileYdimThreadSharedMemIdx);

                if (dTileYdimThreadIdx == dTileXdimThreadIdx) {
                  d_val = exp_g(sub_g(i_val, m_val));
                } else {
                  // (dTileYdimThreadIdx > dTileXdimThreadIdx)
                  // (-> in lower triangular part)
                  scalar_t dtilde_val =
                      float2type<scalar_t>(add_g(f_acc, type2float(i_val)));
                  d_val = exp_g(sub_g(dtilde_val, m_val));
                }
                deltaDtilde_val = mul_g(deltaD_val, d_val);
              }
              // store the D'Tile entries in dstrRTile
              SMEMARRAY(dstrRTile, KVtileDim, dTileYdimThreadSharedMemIdx,
                        dTileXdimThreadSharedMemIdx) = d_val;
              // store the deltaDtildeTile entries in dDTile
              SMEMARRAY(dDTile, KVtileDim, dTileYdimThreadSharedMemIdx,
                        dTileXdimThreadSharedMemIdx) = deltaDtilde_val;

            } // end for (dTileYdimThreadSharedMemIdx)
          }   // end if (dTileXdimThreadSharedMemIdx < KVtileDim)
        }     // end for (dTileXdimIdx)
        __syncthreads();

#ifdef OUTPUTDTile
        //! DEBUG: write D'tile to global memory
        // left upper corner of cWarpTileBlock in C (global memory)
        //* cdTile Global Memory Index (Debug only)
        const uint cdTileGridXYGlobalMemIdx =
            batchHeadGridXGlobalMemIdxCD + (seqLen * QtileDim) * qTileIdx;
        const uint cdTileBlockGlobalMemIdx =
            cdTileGridXYGlobalMemIdx + (kvTileIdx * KVtileDim * gridDim.y) +
            (1 * KVtileDim) * blockIdx.y;

        const uint cdWarpTileYEnd = CEIL_DIV(QtileDim, blockDim.y);
        const uint cdWarpTileXEnd = CEIL_DIV(KVtileDim, blockDim.x);
        for (uint cdWarpTileYIdx = 0; cdWarpTileYIdx < cdWarpTileYEnd;
             ++cdWarpTileYIdx) {
          for (uint cdWarpTileXIdx = 0; cdWarpTileXIdx < cdWarpTileXEnd;
               ++cdWarpTileXIdx) {
            //? cTileIdxes
            //* shared memory:
            const uint cdWarpTileThreadSharedMemYIdx =
                blockDim.y * cdWarpTileYIdx + threadIdx.y;
            const uint cdWarpTileThreadSharedMemXIdx =
                blockDim.x * cdWarpTileXIdx + threadIdx.x;
            //* global memory:
            const uint cdWarpTileBlockGlobalMemIdx =
                cdTileBlockGlobalMemIdx +
                (seqLen * blockDim.y) * cdWarpTileYIdx +
                blockDim.x * cdWarpTileXIdx;
            const uint cdWarpTileThreadGlobalMemIdx =
                cdWarpTileBlockGlobalMemIdx + seqLen * threadIdx.y +
                threadIdx.x;

            matC[cdWarpTileThreadGlobalMemIdx] =
                SMEMARRAY(dstrRTile, KVtileDim, cdWarpTileThreadSharedMemYIdx,
                          cdWarpTileThreadSharedMemXIdx);
          }
        }
#endif
        //! Compute deltaDtildeTile = deltaDTile * D'Tile
        // Computed with pointwise multiplication in the previous step

#ifdef OUTPUTdDtildeTile
        //! DEBUG: write Dtilde Tile to global memory
        // left upper corner of cWarpTileBlock in C (global memory)
        //* cdTile Global Memory Index (Debug only)
        const uint cdTileGridXYGlobalMemIdx =
            batchHeadGridXGlobalMemIdxCD + (seqLen * QtileDim) * qTileIdx;
        const uint cdTileBlockGlobalMemIdx =
            cdTileGridXYGlobalMemIdx + (kvTileIdx * KVtileDim * gridDim.y) +
            (1 * KVtileDim) * blockIdx.y;

        const uint cdWarpTileYEnd = CEIL_DIV(QtileDim, blockDim.y);
        const uint cdWarpTileXEnd = CEIL_DIV(KVtileDim, blockDim.x);
        for (uint cdWarpTileYIdx = 0; cdWarpTileYIdx < cdWarpTileYEnd;
             ++cdWarpTileYIdx) {
          for (uint cdWarpTileXIdx = 0; cdWarpTileXIdx < cdWarpTileXEnd;
               ++cdWarpTileXIdx) {
            //? cTileIdxes
            //* shared memory:
            const uint cdWarpTileThreadSharedMemYIdx =
                blockDim.y * cdWarpTileYIdx + threadIdx.y;
            const uint cdWarpTileThreadSharedMemXIdx =
                blockDim.x * cdWarpTileXIdx + threadIdx.x;
            //* global memory:
            const uint cdWarpTileBlockGlobalMemIdx =
                cdTileBlockGlobalMemIdx +
                (seqLen * blockDim.y) * cdWarpTileYIdx +
                blockDim.x * cdWarpTileXIdx;
            const uint cdWarpTileThreadGlobalMemIdx =
                cdWarpTileBlockGlobalMemIdx + seqLen * threadIdx.y +
                threadIdx.x;

            matC[cdWarpTileThreadGlobalMemIdx] =
                SMEMARRAY(dDTile, KVtileDim, cdWarpTileThreadSharedMemYIdx,
                          cdWarpTileThreadSharedMemXIdx);
          }
        }
#endif

        //! Compute pTile = deltaCTile * D'Tile
        //! Compute rTile = sTile * D'Tile
        // left upper corner of cWarpTileBlock in C (global memory)
        const uint prWarpTileYEnd = CEIL_DIV(QtileDim, blockDim.y);
        const uint prWarpTileXEnd = CEIL_DIV(KVtileDim, blockDim.x);
        for (uint prWarpTileYIdx = 0; prWarpTileYIdx < prWarpTileYEnd;
             ++prWarpTileYIdx) {
          for (uint prWarpTileXIdx = 0; prWarpTileXIdx < prWarpTileXEnd;
               ++prWarpTileXIdx) {
            //? cTileIdxes
            //* shared memory:
            const uint prWarpTileThreadSharedMemYIdx =
                blockDim.y * prWarpTileYIdx + threadIdx.y;
            const uint prWarpTileThreadSharedMemXIdx =
                blockDim.x * prWarpTileXIdx + threadIdx.x;

            // loading from shared memory
            scalar_t deltaC_val =
                SMEMARRAY(dCDcsTile, KVtileDim, prWarpTileThreadSharedMemYIdx,
                          prWarpTileThreadSharedMemXIdx);
            scalar_t dstr_val =
                SMEMARRAY(dstrRTile, KVtileDim, prWarpTileThreadSharedMemYIdx,
                          prWarpTileThreadSharedMemXIdx);
            scalar_t s_val =
                SMEMARRAY(sPTile, KVtileDim, prWarpTileThreadSharedMemYIdx,
                          prWarpTileThreadSharedMemXIdx);

            // pointwise operations
            scalar_t p_val = mul_g(deltaC_val, dstr_val);
            scalar_t r_val = mul_g(s_val, dstr_val);

            // store in shared memory
            SMEMARRAY(sPTile, KVtileDim, prWarpTileThreadSharedMemYIdx,
                      prWarpTileThreadSharedMemXIdx) = p_val;
            SMEMARRAY(dstrRTile, KVtileDim, prWarpTileThreadSharedMemYIdx,
                      prWarpTileThreadSharedMemXIdx) = r_val;
          }
        }

#ifdef OUTPUTPRTile
        //! DEBUG: write P or R Tile to global memory
        // left upper corner of cWarpTileBlock in C (global memory)
        //* cdTile Global Memory Index (Debug only)
        const uint cdTileGridXYGlobalMemIdx =
            batchHeadGridXGlobalMemIdxCD + (seqLen * QtileDim) * qTileIdx;
        const uint cdTileBlockGlobalMemIdx =
            cdTileGridXYGlobalMemIdx + (kvTileIdx * KVtileDim * gridDim.y) +
            (1 * KVtileDim) * blockIdx.y;

        const uint cdWarpTileYEnd = CEIL_DIV(QtileDim, blockDim.y);
        const uint cdWarpTileXEnd = CEIL_DIV(KVtileDim, blockDim.x);
        for (uint cdWarpTileYIdx = 0; cdWarpTileYIdx < cdWarpTileYEnd;
             ++cdWarpTileYIdx) {
          for (uint cdWarpTileXIdx = 0; cdWarpTileXIdx < cdWarpTileXEnd;
               ++cdWarpTileXIdx) {
            //? cTileIdxes
            //* shared memory:
            const uint cdWarpTileThreadSharedMemYIdx =
                blockDim.y * cdWarpTileYIdx + threadIdx.y;
            const uint cdWarpTileThreadSharedMemXIdx =
                blockDim.x * cdWarpTileXIdx + threadIdx.x;
            //* global memory:
            const uint cdWarpTileBlockGlobalMemIdx =
                cdTileBlockGlobalMemIdx +
                (seqLen * blockDim.y) * cdWarpTileYIdx +
                blockDim.x * cdWarpTileXIdx;
            const uint cdWarpTileThreadGlobalMemIdx =
                cdWarpTileBlockGlobalMemIdx + seqLen * threadIdx.y +
                threadIdx.x;
#ifdef OUTPUTPRTileR
            matC[cdWarpTileThreadGlobalMemIdx] =
                SMEMARRAY(dstrRTile, KVtileDim, cdWarpTileThreadSharedMemYIdx,
                          cdWarpTileThreadSharedMemXIdx);
#else
            matC[cdWarpTileThreadGlobalMemIdx] =
                SMEMARRAY(sPTile, KVtileDim, cdWarpTileThreadSharedMemYIdx,
                          cdWarpTileThreadSharedMemXIdx);
#endif
          }
        }
        __syncthreads();
#endif

        //! Compute csDTile = cumsum(deltaDtildeTile) (store in dCDcsTile)
        // TODO extend this with sync between thread blocks over
        // TODO fgate delta computation not implemented yet

        //* 1) Init deltaDcsIterHBM to zero in HBM
        // deltaDcsIterHBM: (gridDim.y x QTileDim)
        const uint deltaDcsChunkArrGridXYGlobalMemIdx =
            batchHeadGridXGlobalMemIdxDeltaDcsChunkArr + (QtileDim)*blockIdx.y;
        // we flatten the threads to 1D along the y-dim (qTileDim)
        // in this way we make sure that global mem access is coalesced
        const uint deltaDcsChunkArrQTileDirEnd =
            CEIL_DIV(QtileDim, blockDim.x * blockDim.y);
        for (uint deltaDcsChunkArrQTileDirIdx = 0;
             deltaDcsChunkArrQTileDirIdx < deltaDcsChunkArrQTileDirEnd;
             ++deltaDcsChunkArrQTileDirIdx) {
          //? deltaDcsIterHBM idxes
          //* TB local idx
          const uint deltaDcsChunkArrQTileDirThreadSharedMemIdx =
              flatThreadIdx +
              blockDim.x * blockDim.y * deltaDcsChunkArrQTileDirIdx;
          //* global memory
          const uint deltaDcsIterHBMXYQtileGlobalMemIdx =
              deltaDcsChunkArrGridXYGlobalMemIdx +
              deltaDcsChunkArrQTileDirThreadSharedMemIdx;

          if (deltaDcsChunkArrQTileDirThreadSharedMemIdx < QtileDim) {
            // set to 0.0f
            csDeltaDTildeChunkArr[deltaDcsIterHBMXYQtileGlobalMemIdx] =
                dscalar_zero<float>();
          }
        }
        __syncthreads();

        //* 1a) Calculate local cumsum along the j-direction (kvTileDim / x-dim)
        //* 1b) If last cumsum tile col is at TB boundary (not at main
        //* diagonal!), write to deltaDcsIterHBM[gridDim.y], sync TBs
        // loop in i-direction (qTileDim / y-dim)
        const uint csDTileYdimEnd = CEIL_DIV(QtileDim, blockDim.x * blockDim.y);
        for (uint csDtileYdimThreadSharedMemIdx = 0;
             csDtileYdimThreadSharedMemIdx < csDTileYdimEnd;
             ++csDtileYdimThreadSharedMemIdx) {
          //? csDTile idxes
          //* shared memory
          const uint csDTileYdimThreadSharedMemIdx =
              flatThreadIdx +
              blockDim.x * blockDim.y * csDtileYdimThreadSharedMemIdx;

          //* csDTile global index (y-dim / QtileDim) (virtual, as never
          // materialized fully)
          const uint csDTileYdimThreadIdx =
              sTileYdimBlockYIdx + csDTileYdimThreadSharedMemIdx;

          if (csDTileYdimThreadSharedMemIdx < QtileDim) {
            float acc = 0.0f;
            bool reachedMainDiagonal = false;
            for (uint csDTileXdimThreadSharedMemIdx = 0;
                 csDTileXdimThreadSharedMemIdx < KVtileDim;
                 ++csDTileXdimThreadSharedMemIdx) {
              //* csDtile global index (x-dim / KVtiledim) (virtual, as never
              // materialized fully)
              const uint csDTileXdimThreadIdx =
                  sTileXdimBlockYIdx + csDTileXdimThreadSharedMemIdx;
              scalar_t dcs_val = dscalar_zero<scalar_t>();
              if (csDTileYdimThreadIdx > csDTileXdimThreadIdx) {
                scalar_t d_val =
                    SMEMARRAY(dDTile, KVtileDim, csDTileYdimThreadSharedMemIdx,
                              csDTileXdimThreadSharedMemIdx);
                acc = add_g(acc, type2float(d_val));
                dcs_val = float2type<scalar_t>(acc);
              } else {
                reachedMainDiagonal = true;
              }
              SMEMARRAY(dCDcsTile, KVtileDim, csDTileYdimThreadSharedMemIdx,
                        csDTileXdimThreadSharedMemIdx) = dcs_val;
            }
            if (!reachedMainDiagonal) {
              // write to global memory
              const uint deltaDcsChunkArrThreadGlobalMemIdx =
                  deltaDcsChunkArrGridXYGlobalMemIdx +
                  csDTileYdimThreadSharedMemIdx;
              csDeltaDTildeChunkArr[deltaDcsChunkArrThreadGlobalMemIdx] = acc;
            }
          }
        }
        __syncthreads();
        gridGroup.sync();

        //* 2) Build the cumsum correction per TB: deltaDcsLoopHBM +
        // sum(deltaDcsIterHBM[<blockIdx.y])
        // store the result in mDeltaDcsCorrChunk in SRAM
        // loop in i-direction (qTileDim / y-dim) with flattened threads
        for (uint csDtileYdimThreadSharedMemIdx = 0;
             csDtileYdimThreadSharedMemIdx < csDTileYdimEnd;
             ++csDtileYdimThreadSharedMemIdx) {
          //? csDTile idxes
          //* shared memory
          const uint csDTileYdimThreadSharedMemIdx =
              flatThreadIdx +
              blockDim.x * blockDim.y * csDtileYdimThreadSharedMemIdx;

          if (csDTileYdimThreadSharedMemIdx < QtileDim) {
            // global memory index for the cumsum correction
            const uint csDeltaDTildeVecThreadGlobalMemIdx =
                nmfChunkBlockGlobalMemIdx + csDTileYdimThreadSharedMemIdx;
            // load the global cumsum correction from HBM
            // and init the total cumusum correction in mDeltaDcsCorrChunk
            float total_cumsum_corr =
                csDeltaDTildeVec[csDeltaDTildeVecThreadGlobalMemIdx];

            // loop over gridDim.y and sum up the cumsum corrections
            // of the other thread blocks with lower blockIdx.y
            for (uint blockIdxY = 0; blockIdxY < blockIdx.y; ++blockIdxY) {
              const uint deltaDcsChunkArrThreadGlobalMemIdx =
                  batchHeadGridXGlobalMemIdxDeltaDcsChunkArr +
                  QtileDim * blockIdxY + csDTileYdimThreadSharedMemIdx;
              total_cumsum_corr = add_g(
                  total_cumsum_corr,
                  csDeltaDTildeChunkArr[deltaDcsChunkArrThreadGlobalMemIdx]);

#ifdef DEBUG_DeltaDCS0
              if ((blockIdx.x == 0) && (blockIdx.y == 1) && (iIdx <= jIdx) &&
                  (flatThreadIdx <= 8)) {
                printf("blockIdx(x,y)=(%d,%d), FTIdx=%d, ijIdx(i,j)=(%d,%d), "
                       "sTileXYIdx(x,y)=(%d,%d), csDYdimSMIdx=%d, bIdxY=%d, "
                       "tot_cs_cor=%f\n",
                       blockIdx.x, blockIdx.y, flatThreadIdx, iIdx, jIdx,
                       sTileXdimBlockYIdx, sTileYdimBlockYIdx,
                       csDTileYdimThreadSharedMemIdx, blockIdxY,
                       total_cumsum_corr);
              }
#endif
            }
            // store the total cumsum correction in mDeltaDcsCorrChunk
            SMEMVECTOR(mDeltaDcsCorrChunk, csDTileYdimThreadSharedMemIdx) =
                float2type<scalar_t>(total_cumsum_corr);

#ifdef DEBUG_DeltaDCS1
            if ((blockIdx.x == 0) && (blockIdx.y <= 1) && (jIdx == 0) &&
                (flatThreadIdx <= 8)) {
              printf("blockIdx(x,y)=(%d,%d), FTIdx=%d, ijIdx(i,j)=(%d,%d), "
                     "sTileXYIdx(x,y)=(%d,%d), csDYdimSMIdx=%d, "
                     "tot_cs_cor=%f\n",
                     blockIdx.x, blockIdx.y, flatThreadIdx, iIdx, jIdx,
                     sTileXdimBlockYIdx, sTileYdimBlockYIdx,
                     csDTileYdimThreadSharedMemIdx, total_cumsum_corr);
            }
#endif
          }
        }

        //* 3a) Add cumsum correction to local cumsum in DcsTile
        //* 3b) Store the cumsum result of the last col of the TB closest to
        //*     the main diagonal (but not at the main diagonal!) in
        //*     deltaDcsLoopHBM
        // outer loop: in j-direction (qTileDim / y-dim) with flattened threads
        // inner loop: in i-direction (kvTileDim / x-dim) per thread
        for (uint csDtileYdimThreadSharedMemIdx = 0;
             csDtileYdimThreadSharedMemIdx < csDTileYdimEnd;
             ++csDtileYdimThreadSharedMemIdx) {
          //? csDTile idxes
          //* shared memory
          const uint csDTileYdimThreadSharedMemIdx =
              flatThreadIdx +
              blockDim.x * blockDim.y * csDtileYdimThreadSharedMemIdx;
          //* csDTile global index (y-dim / QtileDim) (virtual, as never
          // materialized fully)
          const uint csDTileYdimThreadIdx =
              sTileYdimBlockYIdx + csDTileYdimThreadSharedMemIdx;

          if (csDTileYdimThreadSharedMemIdx < QtileDim) {
            const uint csDeltaDTildeVecThreadGlobalMemIdx =
                nmfChunkBlockGlobalMemIdx + csDTileYdimThreadSharedMemIdx;

            // load the total cumsum correction from SRAM
            scalar_t total_cumsum_corr =
                SMEMVECTOR(mDeltaDcsCorrChunk, csDTileYdimThreadSharedMemIdx);

            bool reachedMainDiagonal = false;
            scalar_t dcs_corr_val = dscalar_zero<scalar_t>();

            for (uint csDTileXdimThreadSharedMemIdx = 0;
                 csDTileXdimThreadSharedMemIdx < KVtileDim;
                 ++csDTileXdimThreadSharedMemIdx) {

              //* csDtile global index (x-dim / KVtiledim) (virtual, as never
              // materialized fully)
              const uint csDTileXdimThreadIdx =
                  sTileXdimBlockYIdx + csDTileXdimThreadSharedMemIdx;

              if (csDTileYdimThreadIdx > csDTileXdimThreadIdx) {
                // below main diagonal
                // load&update the cumsum(Dtilde) val
                dcs_corr_val = SMEMARRAY(dCDcsTile, KVtileDim,
                                         csDTileYdimThreadSharedMemIdx,
                                         csDTileXdimThreadSharedMemIdx);

                dcs_corr_val = add_g(dcs_corr_val, total_cumsum_corr);

                // store the updated cumsum(Dtilde) val
                SMEMARRAY(dCDcsTile, KVtileDim, csDTileYdimThreadSharedMemIdx,
                          csDTileXdimThreadSharedMemIdx) = dcs_corr_val;
              } else {
                reachedMainDiagonal = true;
                break;
              }
            }
#ifdef DEBUG_DeltaDCS2
            if ((blockIdx.x == 0) && (blockIdx.y == gridDim.y - 1) &&
                (iIdx >= jIdx) && (flatThreadIdx <= 8)) {
              printf("blockIdx(x,y)=(%d,%d), FTIdx=%d, ijIdx(i,j)=(%d,%d), "
                     "sTileXYIdx(x,y)=(%d,%d), csDYdimGBIdx=%d, "
                     "dcs_cv=%f, reachedMD=%d\n",
                     blockIdx.x, blockIdx.y, flatThreadIdx, iIdx, jIdx,
                     sTileXdimBlockYIdx, sTileYdimBlockYIdx,
                     csDTileYdimThreadIdx, type2float(dcs_corr_val),
                     reachedMainDiagonal);
            }
#endif
            if ((!reachedMainDiagonal) && (iIdx >= jIdx) &&
                (blockIdx.y == gridDim.y - 1)) {
              // onle the thread block closest to the main diagonal
              // but not at the main diagonal should write to global memory
              csDeltaDTildeVec[csDeltaDTildeVecThreadGlobalMemIdx] =
                  type2float(dcs_corr_val);
            }
          }
        }

#ifdef OUTPUTDcsTile
        //! DEBUG: write cumsum(Dtilde) Tile to global memory
        // left upper corner of cWarpTileBlock in C (global memory)
        //* cdTile Global Memory Index (Debug only)
        const uint cdTileGridXYGlobalMemIdx =
            batchHeadGridXGlobalMemIdxCD + (seqLen * QtileDim) * qTileIdx;
        const uint cdTileBlockGlobalMemIdx =
            cdTileGridXYGlobalMemIdx + (kvTileIdx * KVtileDim * gridDim.y) +
            (1 * KVtileDim) * blockIdx.y;

        const uint cdWarpTileYEnd = CEIL_DIV(QtileDim, blockDim.y);
        const uint cdWarpTileXEnd = CEIL_DIV(KVtileDim, blockDim.x);
        for (uint cdWarpTileYIdx = 0; cdWarpTileYIdx < cdWarpTileYEnd;
             ++cdWarpTileYIdx) {
          for (uint cdWarpTileXIdx = 0; cdWarpTileXIdx < cdWarpTileXEnd;
               ++cdWarpTileXIdx) {
            //? cTileIdxes
            //* shared memory:
            const uint cdWarpTileThreadSharedMemYIdx =
                blockDim.y * cdWarpTileYIdx + threadIdx.y;
            const uint cdWarpTileThreadSharedMemXIdx =
                blockDim.x * cdWarpTileXIdx + threadIdx.x;
            //* global memory:
            const uint cdWarpTileBlockGlobalMemIdx =
                cdTileBlockGlobalMemIdx +
                (seqLen * blockDim.y) * cdWarpTileYIdx +
                blockDim.x * cdWarpTileXIdx;
            const uint cdWarpTileThreadGlobalMemIdx =
                cdWarpTileBlockGlobalMemIdx + seqLen * threadIdx.y +
                threadIdx.x;

            matC[cdWarpTileThreadGlobalMemIdx] =
                SMEMARRAY(dCDcsTile, KVtileDim, cdWarpTileThreadSharedMemYIdx,
                          cdWarpTileThreadSharedMemXIdx);
          }
        }
#endif

        //! sum up deltaIChunk & deltaFChunk and update in SRAM
        // sum along i-direction (qTileDim / y-dim)
        // loop in j-direction (kvTileDim / x-dim)
        const uint csDtileXdimXdimEnd =
            CEIL_DIV(KVtileDim, blockDim.x * blockDim.y);
        for (uint csDtileXdimThreadSharedMemIdx = 0;
             csDtileXdimThreadSharedMemIdx < csDtileXdimXdimEnd;
             ++csDtileXdimThreadSharedMemIdx) {
          //? dIdFChunk idxes
          //* shared memory
          const uint dIdFChunkXdimThreadSharedMemIdx =
              flatThreadIdx +
              blockDim.x * blockDim.y * csDtileXdimThreadSharedMemIdx;

          //* dTile global index (x-dim / kvTileDim) (virtual, as never
          // materialized fully)
          const uint dTileXdimThreadIdx =
              sTileXdimBlockYIdx + dIdFChunkXdimThreadSharedMemIdx;

          if (dIdFChunkXdimThreadSharedMemIdx < KVtileDim) {
            float acc_deltaI = 0.0f;
            float acc_deltaF = 0.0f;
            for (uint csDtileYdimThreadSharedMemIdx = 0;
                 csDtileYdimThreadSharedMemIdx < QtileDim;
                 ++csDtileYdimThreadSharedMemIdx) {

              //* dTile global index (y-dim / QtileDim) (virtual, as never
              // materialized fully)
              const uint dTileYdimThreadIdx =
                  sTileYdimBlockYIdx + csDtileYdimThreadSharedMemIdx;
#ifdef DEBUG_deltaISUM0
              if ((blockIdx.x == 0) && (blockIdx.y == 0) &&
                  (flatThreadIdx < 1)) {
                printf("!qTileIdx=%d, kvTileIdx=%d, dTileXdimThreadIdx=%d, "
                       "dIdFChunkXdimTSMIdx=%d, flatTidx=%d, "
                       "tbIdxXY=(%d,%d): csDtileIdx=%d, dTileXdimTIdx=%d, "
                       "dTileYdimTIdx=%d\n",
                       qTileIdx, kvTileIdx, dTileXdimThreadIdx,
                       dIdFChunkXdimThreadSharedMemIdx, flatThreadIdx,
                       threadIdx.x, threadIdx.y, csDtileYdimThreadSharedMemIdx,
                       dTileXdimThreadIdx, dTileYdimThreadIdx);
              }
#endif
              // sum up deltaIChunk
              if (dTileYdimThreadIdx >= dTileXdimThreadIdx) {
                //? sum the entries in deltaDtildeTile
                scalar_t deltaI_val =
                    SMEMARRAY(dDTile, KVtileDim, csDtileYdimThreadSharedMemIdx,
                              dIdFChunkXdimThreadSharedMemIdx);
                acc_deltaI = add_g(acc_deltaI, type2float(deltaI_val));
#ifdef DEBUG_deltaISUM1
                if ((blockIdx.x == 0) && (blockIdx.y == 0) &&
                    (flatThreadIdx < 1)) {
                  printf("qTileIdx=%d, kvTileIdx=%d, dTileXdimThreadIdx=%d, "
                         "dIdFChunkXdimTSMIdx=%d, flatTidx=%d, "
                         "tbIdxXY=(%d,%d), csDtileIdx=%d: "
                         "acc_deltaI=%f\n",
                         qTileIdx, kvTileIdx, dTileXdimThreadIdx,
                         dIdFChunkXdimThreadSharedMemIdx, flatThreadIdx,
                         threadIdx.x, threadIdx.y,
                         csDtileYdimThreadSharedMemIdx, type2float(deltaI_val));
                }
#endif
              }

              // sum up deltaFChunk
              if (dTileYdimThreadIdx > dTileXdimThreadIdx) {
                //? sum the entries in DcsTile
                scalar_t deltaF_val = SMEMARRAY(
                    dCDcsTile, KVtileDim, csDtileYdimThreadSharedMemIdx,
                    dIdFChunkXdimThreadSharedMemIdx);
                acc_deltaF = add_g(acc_deltaF, type2float(deltaF_val));
              }

            } // end for (csDtileYdimThreadSharedMemIdx)

#ifdef DEBUG_deltaISUM2
            if ((blockIdx.x == 0) && (blockIdx.y == 0) && (flatThreadIdx < 1)) {
              printf("qTileIdx=%d, kvTileIdx=%d, dTileXdimThreadIdx=%d, "
                     "dIdFChunkXdimThreadSharedMemIdx=%d, flatTidx=%d, "
                     "tbIdxXY=(%d,%d): "
                     "acc_deltaI=%f\n",
                     qTileIdx, kvTileIdx, dTileXdimThreadIdx,
                     dIdFChunkXdimThreadSharedMemIdx, flatThreadIdx,
                     threadIdx.x, threadIdx.y, acc_deltaI);
            }
#endif

            // update deltaIChunk & deltaFChunk in SMEM
            scalar_t deltaI_val =
                SMEMVECTOR(deltaIChunk, dIdFChunkXdimThreadSharedMemIdx);
            SMEMVECTOR(deltaIChunk, dIdFChunkXdimThreadSharedMemIdx) =
                float2type<scalar_t>(add_g(type2float(deltaI_val), acc_deltaI));

            scalar_t deltaFbar_val =
                SMEMVECTOR(deltaFChunk, dIdFChunkXdimThreadSharedMemIdx);
            SMEMVECTOR(deltaFChunk, dIdFChunkXdimThreadSharedMemIdx) =
                float2type<scalar_t>(
                    add_g(type2float(deltaFbar_val), acc_deltaF));
          } // end if (dIdFChunkXdimThreadSharedMemIdx < KVtileDim)
        }   // end for (csDtileXdimThreadSharedMemIdx)

        //! Compute deltaQTile = pTile  (kTile/sqrt(d))
        // (QtileDim x dimHeads) = (QtileDim x KVtileDim) x (KVtileDim x
        // dimHeads)
        // loops over deltaQTile rows (outer) and columns (inner)
        const uint deltaQWarpTileYEnd = CEIL_DIV(QtileDim, blockDim.y); // rows
        const uint deltaQWarpTileXEnd = CEIL_DIV(dimHeads, blockDim.x); // cols
        for (uint deltaQWarpTileYIdx = 0;
             deltaQWarpTileYIdx < deltaQWarpTileYEnd; ++deltaQWarpTileYIdx) {
          for (uint deltaQWarpTileXIdx = 0;
               deltaQWarpTileXIdx < deltaQWarpTileXEnd; ++deltaQWarpTileXIdx) {
            //? cTileIdxes
            //* shared memory:
            const uint deltaQWarpTileThreadSharedMemYIdx =
                blockDim.y * deltaQWarpTileYIdx + threadIdx.y;
            const uint deltaQWarpTileThreadSharedMemXIdx =
                blockDim.x * deltaQWarpTileXIdx + threadIdx.x;

            // scalar_t qk_acc = dscalar_zero<scalar_t>();
            float acc = 0.0f;
            for (uint i = 0; i < KVtileDim; ++i) {
              acc = add_g(acc,
                          type2float(mul_g(
                              SMEMARRAY(sPTile, KVtileDim,
                                        deltaQWarpTileThreadSharedMemYIdx, i),
                              SMEMARRAY(kTile, dimHeads, i,
                                        deltaQWarpTileThreadSharedMemXIdx))));
            }

            // compute deltaQTile
            scalar_t deltaQ_val =
                float2type<scalar_t>(mul_g(acc, rsqrtf(type2float(dimHeads))));
            SMEMARRAY(deltaQTile, dimHeads, deltaQWarpTileThreadSharedMemYIdx,
                      deltaQWarpTileThreadSharedMemXIdx) = deltaQ_val;
          }
        }
        __syncthreads();

        gridGroup.sync();

        //! Atomic add deltaQTile to deltaQ in HBM
        // We sum up the deltaQTiles in the different thread blocks in the
        // global memory loops over rows (outer) and columns (inner) of
        // deltaQTile
        for (uint deltaQWarpTileYIdx = 0;
             deltaQWarpTileYIdx < deltaQWarpTileYEnd; ++deltaQWarpTileYIdx) {
          for (uint deltaQWarpTileXIdx = 0;
               deltaQWarpTileXIdx < deltaQWarpTileXEnd; ++deltaQWarpTileXIdx) {

            //? deltaQWarpTileIdxes for deltaQ-tile
            //* shared memory:
            const uint deltaQWarpTileThreadSharedMemYIdx =
                blockDim.y * deltaQWarpTileYIdx + threadIdx.y;
            const uint deltaQWarpTileThreadSharedMemXIdx =
                blockDim.x * deltaQWarpTileXIdx + threadIdx.x;

            //* global memory:
            // left upper corner of qTileBlock in Q (global memory)
            const uint deltaQWarpTileBlockGlobalMemIdx =
                qdHdQTileBlockGlobalMemIdx +
                (dimHeads * blockDim.y) * deltaQWarpTileYIdx +
                blockDim.x * deltaQWarpTileXIdx;
            const uint deltaQWarpTileThreadGlobalMemIdx =
                deltaQWarpTileBlockGlobalMemIdx + dimHeads * threadIdx.y +
                threadIdx.x;

            // load deltaQ val from shared memory
            scalar_t deltaQ_val = SMEMARRAY(deltaQTile, dimHeads,
                                            deltaQWarpTileThreadSharedMemYIdx,
                                            deltaQWarpTileThreadSharedMemXIdx);

            // atomic add to deltaQ in HBM
            atomicAdd(&(deltaQ[deltaQWarpTileThreadGlobalMemIdx]), deltaQ_val);
          }
        }
        __syncthreads();

        //! Compute deltaKTile = pTile^T  (qTile/sqrt(d)) and update in SRAM
        // (KVtileDim x dimHeads) = (KVtileDim x QtileDim) x (QtileDim x
        // dimHeads)
        // loops over deltaKTile rows (outer) and columns (inner)
        const uint deltaKWarpTileYEnd = CEIL_DIV(KVtileDim, blockDim.y); // rows
        const uint deltaKWarpTileXEnd = CEIL_DIV(dimHeads, blockDim.x);  // cols
        for (uint deltaKWarpTileYIdx = 0;
             deltaKWarpTileYIdx < deltaKWarpTileYEnd; ++deltaKWarpTileYIdx) {
          for (uint deltaKWarpTileXIdx = 0;
               deltaKWarpTileXIdx < deltaKWarpTileXEnd; ++deltaKWarpTileXIdx) {
            //? cTileIdxes
            //* shared memory:
            const uint deltaKWarpTileThreadSharedMemYIdx =
                blockDim.y * deltaKWarpTileYIdx + threadIdx.y;
            const uint deltaKWarpTileThreadSharedMemXIdx =
                blockDim.x * deltaKWarpTileXIdx + threadIdx.x;

            // scalar_t qk_acc = dscalar_zero<scalar_t>();
            float acc = 0.0f;
            for (uint i = 0; i < QtileDim; ++i) {
              acc = add_g(acc,
                          type2float(mul_g(
                              SMEMARRAY(sPTile, KVtileDim, i,
                                        deltaKWarpTileThreadSharedMemYIdx),
                              SMEMARRAY(qTile, dimHeads, i,
                                        deltaKWarpTileThreadSharedMemXIdx))));
            }

            // compute deltaKTile
            scalar_t deltaK_val_new =
                float2type<scalar_t>(mul_g(acc, rsqrtf(type2float(dimHeads))));

            // update deltaKTile in shared memory
            scalar_t deltaK_val_old = SMEMARRAY(
                deltaKTile, dimHeads, deltaKWarpTileThreadSharedMemYIdx,
                deltaKWarpTileThreadSharedMemXIdx);

            SMEMARRAY(deltaKTile, dimHeads, deltaKWarpTileThreadSharedMemYIdx,
                      deltaKWarpTileThreadSharedMemXIdx) =
                add_g(deltaK_val_old, deltaK_val_new);
          }
        }
        __syncthreads();

        gridGroup.sync();

        //! Compute deltaVTile = rTile^T  (deltaHTile * 1 / nChunk) and update
        //! in SRAM
        // (KVtileDim x dimHeads) = (KVtileDim x QtileDim) x (QtileDim x
        // dimHeads)
        // loops over deltaKTile rows (outer) and columns (inner)
        const uint deltaVWarpTileYEnd = CEIL_DIV(KVtileDim, blockDim.y); // rows
        const uint deltaVWarpTileXEnd = CEIL_DIV(dimHeads, blockDim.x);  // cols
        for (uint deltaVWarpTileYIdx = 0;
             deltaVWarpTileYIdx < deltaVWarpTileYEnd; ++deltaVWarpTileYIdx) {
          for (uint deltaVWarpTileXIdx = 0;
               deltaVWarpTileXIdx < deltaVWarpTileXEnd; ++deltaVWarpTileXIdx) {
            //? deltaVTileIdxes
            //* shared memory:
            const uint deltaVWarpTileThreadSharedMemYIdx =
                blockDim.y * deltaVWarpTileYIdx + threadIdx.y;
            const uint deltaVWarpTileThreadSharedMemXIdx =
                blockDim.x * deltaVWarpTileXIdx + threadIdx.x;

            float acc = 0.0f;
            for (uint i = 0; i < QtileDim; ++i) {
              scalar_t r_val = SMEMARRAY(dstrRTile, KVtileDim, i,
                                         deltaVWarpTileThreadSharedMemYIdx);

              scalar_t deltaH_val = SMEMARRAY(
                  deltaHTile, dimHeads, i, deltaVWarpTileThreadSharedMemXIdx);
              scalar_t n_val = SMEMVECTOR(nChunk, i);

              deltaH_val = div_g(deltaH_val, n_val);

              acc = add_g(acc, type2float(mul_g(r_val, deltaH_val)));
            }

            // compute deltaVTile
            scalar_t deltaV_val_new = float2type<scalar_t>(acc);

            // update deltaVTile in shared memory
            scalar_t deltaV_val_old = SMEMARRAY(
                deltaVTile, dimHeads, deltaVWarpTileThreadSharedMemYIdx,
                deltaVWarpTileThreadSharedMemXIdx);

            SMEMARRAY(deltaVTile, dimHeads, deltaVWarpTileThreadSharedMemYIdx,
                      deltaVWarpTileThreadSharedMemXIdx) =
                add_g(deltaV_val_old, deltaV_val_new);
          }
        }
        __syncthreads();

        gridGroup.sync();

      } // end looplevel 2 (i-loop)

      gridGroup.sync();

      //! Store deltaKTile & deltaVTile in HBM
      // loops over rows (outer) and columns (inner) of kTile and vTile
      const uint dKdVWarpTileYEnd = CEIL_DIV(KVtileDim, blockDim.y); // rows
      const uint dKdVWarpTileXEnd = CEIL_DIV(dimHeads, blockDim.x);  // cols
      for (uint dKdVWarpTileYIdx = 0; dKdVWarpTileYIdx < dKdVWarpTileYEnd;
           ++dKdVWarpTileYIdx) {
        for (uint dKdVWarpTileXIdx = 0; dKdVWarpTileXIdx < dKdVWarpTileXEnd;
             ++dKdVWarpTileXIdx) {
          //? dKdVWarpTileIdxes for k-tile AND v-tile
          //* shared memory:
          const uint dKdVWarpTileThreadSharedMemYIdx =
              blockDim.y * dKdVWarpTileYIdx + threadIdx.y;
          const uint dKdVWarpTileThreadSharedMemXIdx =
              blockDim.x * dKdVWarpTileXIdx + threadIdx.x;
          //* global memory:
          // left upper corner of kTileBlock in K (global memory)
          const uint kvWarpTileBlockGlobalMemIdx =
              kvdKdVTileBlockGlobalMemIdx +
              (dimHeads * blockDim.y) * dKdVWarpTileYIdx +
              blockDim.x * dKdVWarpTileXIdx;
          const uint kvWarpTileThreadGlobalMemIdx =
              kvWarpTileBlockGlobalMemIdx + dimHeads * threadIdx.y +
              threadIdx.x;

          // write to HBM
          deltaK[kvWarpTileThreadGlobalMemIdx] =
              SMEMARRAY(deltaKTile, dimHeads, dKdVWarpTileThreadSharedMemYIdx,
                        dKdVWarpTileThreadSharedMemXIdx);
          deltaV[kvWarpTileThreadGlobalMemIdx] =
              SMEMARRAY(deltaVTile, dimHeads, dKdVWarpTileThreadSharedMemYIdx,
                        dKdVWarpTileThreadSharedMemXIdx);

          // clear SRAM for next iteration
          SMEMARRAY(deltaKTile, dimHeads, dKdVWarpTileThreadSharedMemYIdx,
                    dKdVWarpTileThreadSharedMemXIdx) = dscalar_zero<scalar_t>();
          SMEMARRAY(deltaVTile, dimHeads, dKdVWarpTileThreadSharedMemYIdx,
                    dKdVWarpTileThreadSharedMemXIdx) = dscalar_zero<scalar_t>();
        }
      }
      __syncthreads();

      //! Store deltaIChunk & deltaFChunk in HBM
      // loop in j-direction (kvTileDim / x-dim)
      const uint dIdFChunkEnd = CEIL_DIV(KVtileDim, blockDim.x * blockDim.y);
      for (uint dIdFChunkIdx = 0; dIdFChunkIdx < dIdFChunkEnd; ++dIdFChunkIdx) {
        //? dIdFChunk idxes
        //* shared memory
        const uint dIdFChunkThreadSharedMemIdx =
            flatThreadIdx + blockDim.x * blockDim.y * dIdFChunkIdx;
        //* global memory
        const uint dIdFThreadGlobalMemIdx =
            iChunkBlockGlobalMemIdx + flatThreadIdx;

        //* global dFIdx (virtual, as never materialized fully)
        const uint dFChunkXdimBlockYThreadIdx =
            sTileXdimBlockYIdx + dIdFChunkThreadSharedMemIdx;

        if (dIdFChunkThreadSharedMemIdx < KVtileDim) {
          deltaIGatePreact[dIdFThreadGlobalMemIdx] =
              SMEMVECTOR(deltaIChunk, dIdFChunkThreadSharedMemIdx);

          // multiply with sigmoid derivative: sigmoid(-fGatePreact)
          scalar_t deltaFbar_val =
              SMEMVECTOR(deltaFChunk, dIdFChunkThreadSharedMemIdx);

          // avoid accessing out of bounds (need to check that
          // the very last value is not written and not accessed)
          if (dFChunkXdimBlockYThreadIdx < seqLen - 1) {
            // need to shift index since the first forgetgate f_1 is not used
            scalar_t fPreact_val = fGatePreact[dIdFThreadGlobalMemIdx + 1];

            scalar_t deltaF_val =
                mul_g(deltaFbar_val, sigmoid_g(neg_g(fPreact_val)));

            deltaFGatePreact[dIdFThreadGlobalMemIdx + 1] = deltaF_val;
          }
#ifdef DEBUG_WRdeltaI
          if ((blockIdx.x == 0) && (blockIdx.y == 0) && (flatThreadIdx <= 8)) {
            printf("kvTileIdx=%d, dIdFChunkIdx=%d"
                   "iChunkBlockGlobalMemIdx=%d, flatTidx=%d, tbIdxXY=(%d,%d): "
                   "deltaI[%d]=%f\n",
                   kvTileIdx, dIdFChunkIdx, iChunkBlockGlobalMemIdx,
                   flatThreadIdx, threadIdx.x, threadIdx.y,
                   dIdFThreadGlobalMemIdx,
                   type2float(
                       SMEMVECTOR(deltaIChunk, dIdFChunkThreadSharedMemIdx)));
          }
#endif
        }
      }
      __syncthreads();

    } // end looplevel 1 (j-loop)

  } // end looplevel 0
} // kernels::vlstm_fw

template <typename scalar_t>
void kernel_dispatchers::vlstm_bw_dispatch(
    scalar_t *deltaQ, scalar_t *deltaK, scalar_t *deltaV,
    scalar_t *deltaIGatePreact, scalar_t *deltaFGatePreact, scalar_t *matC,
    scalar_t *deltaH, scalar_t *matQ, scalar_t *matK, scalar_t *matV,
    scalar_t *iGatePreact, scalar_t *fGatePreact, scalar_t *vecN,
    scalar_t *vecM, int batchSize, int numHeads, int seqLen, int dimHeads) {
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

  const uint gridDimY = 2;
  const dim3 gridDims(batchSize * numHeads, gridDimY);
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
  const uint ididfChunkSharedMemSize =
      sizeof(scalar_t) * KVtileDim * (1 + SHARED_MEM_PADDING);

  const uint nmfChunkSharedMemSize =
      sizeof(scalar_t) * QtileDim * (1 + SHARED_MEM_PADDING);

  //? intermediate tiles
  const uint sdprdcddTileSharedMemSize =
      sizeof(scalar_t) * QtileDim * (KVtileDim + SHARED_MEM_PADDING);

  // we keep these as float as it acts as accumulator
  // for the fTileRow during D' computation
  const uint fTileRowSharedMemSize =
      sizeof(float) * KVtileDim * (1 + SHARED_MEM_PADDING);

  const uint sharedMemorySize =
      3 * qdQdHTileSharedMemSize + 4 * kvdKdVTileSharedMemSize +
      3 * ididfChunkSharedMemSize + 3 * nmfChunkSharedMemSize +
      4 * sdprdcddTileSharedMemSize + 1 * fTileRowSharedMemSize;

  printf("blocksxy: %d-%d, threadsxy: %d-%d, shared_mem in bytes: %d\n",
         gridDims.x, gridDims.y, blockDims.x, blockDims.y, sharedMemorySize);
  // cudaSetDevice(0);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // TODO bring this back later. For debugging purposes, we need to allocate
  // memory in torch
  //? Allocate intermediate global memory for cumsum(deltaDtildeTile) along
  // KVdim
  //* csDeltaDTildeChunkArr: Used for sync within all y-dim TBs
  uint csDeltaDTildeChunkArrGlobalMemSize =
      sizeof(float) * batchSize * numHeads * QtileDim * gridDimY;
  float *csDeltaDTildeChunkArr;
  gpuErrchk(cudaMalloc((void **)&csDeltaDTildeChunkArr,
                       csDeltaDTildeChunkArrGlobalMemSize));

  //* csDeltaDTildeVec: Used to store previous computations over j - iterations
  //(iterations over the gridDimY)
  uint csDeltaDTildeVecGlobalMemSize =
      sizeof(float) * batchSize * numHeads * seqLen;
  float *csDeltaDTildeVec;
  gpuErrchk(
      cudaMalloc((void **)&csDeltaDTildeVec, csDeltaDTildeVecGlobalMemSize));
  // init the memory to zero
  // cudaMemset only works with integers, but float 0.0 has 0000 0000 in binary,
  // so it should work
  gpuErrchk(cudaMemset(csDeltaDTildeVec, 0, csDeltaDTildeVecGlobalMemSize));

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
                        (void *)&matC,
                        (void *)&deltaH,
                        (void *)&matQ,
                        (void *)&matK,
                        (void *)&matV,
                        (void *)&iGatePreact,
                        (void *)&fGatePreact,
                        (void *)&vecN,
                        (void *)&vecM,
                        (void *)&csDeltaDTildeChunkArr,
                        (void *)&csDeltaDTildeVec,
                        (void *)&batchSize,
                        (void *)&numHeads,
                        (void *)&seqLen,
                        (void *)&dimHeads};

  cudaLaunchCooperativeKernel((void *)kernel, gridDims, blockDims, kernelArgs,
                              sharedMemorySize, stream);

  gpuErrchk(cudaPeekAtLastError());

  // free the allocated memory
  gpuErrchk(cudaFree(csDeltaDTildeChunkArr));
  gpuErrchk(cudaFree(csDeltaDTildeVec));

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
}

// this is needed to make sure that the compiler instantiates the template
template void kernel_dispatchers::vlstm_bw_dispatch<__nv_bfloat16>(
    __nv_bfloat16 *deltaQ, __nv_bfloat16 *deltaK, __nv_bfloat16 *deltaV,
    __nv_bfloat16 *deltaIGatePreact, __nv_bfloat16 *deltaFGatePreact,
    __nv_bfloat16 *matC, __nv_bfloat16 *deltaH, __nv_bfloat16 *matQ,
    __nv_bfloat16 *matK, __nv_bfloat16 *matV, __nv_bfloat16 *iGatePreact,
    __nv_bfloat16 *fGatePreact, __nv_bfloat16 *vecN, __nv_bfloat16 *vecM,
    int batchSize, int numHeads, int seqLen, int dimHeads);
template void kernel_dispatchers::vlstm_bw_dispatch<__half>(
    __half *deltaQ, __half *deltaK, __half *deltaV, __half *deltaIGatePreact,
    __half *deltaFGatePreact, __half *matC, __half *deltaH, __half *matQ,
    __half *matK, __half *matV, __half *iGatePreact, __half *fGatePreact,
    __half *vecN, __half *vecM, int batchSize, int numHeads, int seqLen,
    int dimHeads);
template void kernel_dispatchers::vlstm_bw_dispatch<float>(
    float *deltaQ, float *deltaK, float *deltaV, float *deltaIGatePreact,
    float *deltaFGatePreact, float *matC, float *deltaH, float *matQ,
    float *matK, float *matV, float *iGatePreact, float *fGatePreact,
    float *vecN, float *vecM, int batchSize, int numHeads, int seqLen,
    int dimHeads);
} // namespace vlstm