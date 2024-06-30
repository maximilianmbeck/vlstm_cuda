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
__global__ void vlstm_fw(scalar_t *matH, scalar_t *vecN, scalar_t *vecM,
                         scalar_t *matC, scalar_t *matQ, scalar_t *matK,
                         scalar_t *matV, scalar_t *iGatePreact,
                         scalar_t *fGatePreact, float *fTileColLast,
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
// #define DEBUG8 1
// #define DEBUG9 1
// #define DEBUG10 1
// #define DEBUG11 1
// #define DEBUG12 1

// #define DEBUG_fcolval1 1
// #define DEBUG_fcolval2 1
// #define DEBUG_fcolval3 1

#define OUTPUT_matD 1

/**
Conventions:
- chunk: A 1D vector in shared memory
- tile: A 2D matrix in shared memory
*/

/* vLSTM Forward Kernel v0 */

template <typename scalar_t, int TblockDim, int QtileDim, int KVtileDim>
__global__ void
kernels::vlstm_fw(scalar_t *matH, scalar_t *vecN, scalar_t *vecM,
                  scalar_t *matC, scalar_t *matQ, scalar_t *matK,
                  scalar_t *matV, scalar_t *iGatePreact, scalar_t *fGatePreact,
                  float *fTileColLast, int batchSize, int numHeads, int seqLen,
                  int dimHeads) {
  // int tIdx = threadIdx.x + blockDim.x * threadIdx.y;
#ifdef DEBUG
  if ((blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0) &&
      (threadIdx.y == 0)) {
    printf(
        "In FW-Kernel: gdim.x: %d, gdim.y: %d, gdim.z: %d, bdim.x: %d, bdim.y: "
        "%d\n",
        gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y);
  }
#endif

#ifdef DEBUG
  if ((blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0) &&
      (threadIdx.y == 0)) {
    printf("In FW-Kernel: QtileDim: %d, KVtileDim: %d, TblockDim:%d\n",
           QtileDim, KVtileDim, TblockDim);
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
  // TODO initialize cTile to 0
  scalar_t *cTile =
      (scalar_t *)&vTile[KVtileDim * (dimHeads + SHARED_MEM_PADDING)];
  // init result hTile (QTileDim x dimHeads) in shared memory
  scalar_t *hTile =
      (scalar_t *)&cTile[QtileDim * (KVtileDim + SHARED_MEM_PADDING)];
  // init dTile (QTileDim x KVTileDim) in shared memory for forget and input
  // gate matrix
  scalar_t *dTile =
      (scalar_t *)&hTile[QtileDim * (dimHeads + SHARED_MEM_PADDING)];

  //? for input and forget gate
  // init iChunk (KVTileDim x 1) in shared memory for input gate
  scalar_t *iChunk =
      (scalar_t *)&dTile[QtileDim * (dimHeads + SHARED_MEM_PADDING)];
  // init fChunk (QTileDim x 1) in shared memory for forget gate
  scalar_t *fChunk = (scalar_t *)&iChunk[KVtileDim * (1 + SHARED_MEM_PADDING)];
  // init fTileCol (QTileDim x 1) in shared memory for forget gate (first column
  // of the QtileDim x KVtileDim dTile)
  float *fTileCol = (float *)&fChunk[QtileDim * (1 + SHARED_MEM_PADDING)];

  // init mChunk (QTileDim x 1) in shared memory for max state of
  // dTile
  scalar_t *mChunk = (scalar_t *)&fTileCol[QtileDim * (1 + SHARED_MEM_PADDING)];
  // init mPrevTileCol (QTileDim x 1) in shared memory for previous
  // max state of dTile
  scalar_t *mPrevChunk =
      (scalar_t *)&mChunk[QtileDim * (1 + SHARED_MEM_PADDING)];
  // init lChunk (QTileDim x 1) in shared memory for rowsum of cTile * dTile
  scalar_t *lChunk =
      (scalar_t *)&mPrevChunk[QtileDim * (1 + SHARED_MEM_PADDING)];
  // init lPrevChunk (QTileDim x 1) in shared memory for previous rowsum of
  // cTile * dTile
  scalar_t *lPrevChunk =
      (scalar_t *)&lChunk[QtileDim * (1 + SHARED_MEM_PADDING)];
  // init nChunk (QTileDim x 1) in shared memory for normalizer state
  scalar_t *nChunk =
      (scalar_t *)&lPrevChunk[QtileDim * (1 + SHARED_MEM_PADDING)];
  // init nPrevChunk (QTileDim x 1) in shared memory for previous normalizer
  // state
  scalar_t *nPrevChunk =
      (scalar_t *)&nChunk[QtileDim * (1 + SHARED_MEM_PADDING)];

  //! PARALLELIZE ALONG BATCHSIZE * NUMHEADS (gridDim.x)
  const uint batchHeadStepQKV = seqLen * dimHeads;
  const uint batchHeadStepIFgateNMchunk = seqLen * 1;
  const uint batchHeadStepCD = seqLen * seqLen;
  const uint batchHeadStepFtileColLast = 1;
  const uint numBatchHeads = batchSize * numHeads;
  // End for looplevel 0:
  const uint batchHeadEnd = CEIL_DIV(numBatchHeads, gridDim.x);
  // looplevel 0: loop over batches and heads
  for (uint batchHeadIdx = 0; batchHeadIdx < batchHeadEnd; ++batchHeadIdx) {

    uint batchHeadGridXGlobalMemIdxQKV =
        (batchHeadStepQKV * gridDim.x) * batchHeadIdx +
        (batchHeadStepQKV)*blockIdx.x;

    uint batchHeadGridXGlobalMemIdxIFgateNMchunk =
        (batchHeadStepIFgateNMchunk * gridDim.x) * batchHeadIdx +
        (batchHeadStepIFgateNMchunk)*blockIdx.x;

    uint batchHeadGridXGlobalMemIdxCD =
        (batchHeadStepCD * gridDim.x) * batchHeadIdx +
        (batchHeadStepCD)*blockIdx.x;

    uint batchHeadGridXGlobalMemIdxFtileColLast =
        (batchHeadStepFtileColLast * gridDim.x) * batchHeadIdx +
        (batchHeadStepFtileColLast)*blockIdx.x;

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
    // Ends for looplevel 1:
    const uint qTileEnd = CEIL_DIV(seqLen, QtileDim * gridDim.y);
    // looplevel 1: loop over Qtile blocks along seqLen dim
    for (uint qTileIdx = 0; qTileIdx < qTileEnd; ++qTileIdx) {

      //* qTile Global Memory Index
      // (grid&block) offset in Q matrix for qTile (global memory)
      const uint qTileGridXYGlobalMemIdx =
          batchHeadGridXGlobalMemIdxQKV +
          (dimHeads * QtileDim * gridDim.y) * qTileIdx;
      const uint qTileBlockGlobalMemIdx =
          qTileGridXYGlobalMemIdx + (dimHeads * QtileDim) * blockIdx.y;

      //* cTile Global Memory Index (virtual, as never materialized fully)
      // (grid&block) offset Y-axis in S = Q*K^T matrix (along sequence
      // dimension) (used for checking causality)
      const uint cTileGridYIdx = QtileDim * gridDim.y * qTileIdx;
      const uint cTileBlockYIdx = cTileGridYIdx + QtileDim * blockIdx.y;

      //   //* cdTile Global Memory Index (Debug only)
      //   const uint cdTileGridXYGlobalMemIdx =
      //       batchHeadGridXGlobalMemIdxCD +
      //       (seqLen * QtileDim * gridDim.y) * qTileIdx;
      //   const uint cdTileBlockGlobalMemIdx =
      //       cdTileGridXYGlobalMemIdx + (seqLen * QtileDim) * blockIdx.y;

#ifdef DEBUG5
      if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        printf("B<%d,%d> qTileIdx: %d, qTileEnd: %d, "
               "qTileBlockGlobalMemIdx: "
               "%d, \n",
               blockIdx.x, blockIdx.y, qTileIdx, qTileEnd,
               qTileBlockGlobalMemIdx);
      }
#endif

      //! qTile Loading
      // loops over rows (outer) and columns (inner) of qTile
      const uint qWarpTileYEnd = CEIL_DIV(QtileDim, blockDim.y);
      const uint qWarpTileXEnd = CEIL_DIV(dimHeads, blockDim.x);
      for (uint qWarpTileYIdx = 0; qWarpTileYIdx < qWarpTileYEnd;
           ++qWarpTileYIdx) {
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

      //? flatten the threads to 1D
      const uint flatThreadIdx = blockDim.x * threadIdx.y + threadIdx.x;

      //! init mPrevChunk to -inf,
      //! init lPrevChunk to 0, init nPrevChunk to 0
      // Y: seqLen (or QtileDim), X: 1 (fTileCol has only Y dimension)
      const uint fTileColChunkYEnd =
          CEIL_DIV(QtileDim, blockDim.x * blockDim.y);
      for (uint fTileColChunkYIdx = 0; fTileColChunkYIdx < fTileColChunkYEnd;
           ++fTileColChunkYIdx) {
        //? f idxes
        //* shared memory:
        const uint fThreadSharedMemYIdx =
            flatThreadIdx + blockDim.x * blockDim.y * fTileColChunkYIdx;

        if (fThreadSharedMemYIdx < QtileDim) {
          SMEMVECTOR(fTileCol, fThreadSharedMemYIdx) = 0.0f;
          // mChunk & mPrevChunk
          SMEMVECTOR(mPrevChunk, fThreadSharedMemYIdx) =
              float2type<scalar_t>(-CUDART_INF_F);
          SMEMVECTOR(mChunk, fThreadSharedMemYIdx) =
              float2type<scalar_t>(-CUDART_INF_F);
          // lChunk & lPrevChunk
          SMEMVECTOR(lPrevChunk, fThreadSharedMemYIdx) =
              float2type<scalar_t>(0.0f);
          SMEMVECTOR(lChunk, fThreadSharedMemYIdx) = float2type<scalar_t>(0.0f);
          // nChunk & nPrevChunk
          SMEMVECTOR(nPrevChunk, fThreadSharedMemYIdx) =
              float2type<scalar_t>(0.0f);
          SMEMVECTOR(nChunk, fThreadSharedMemYIdx) = float2type<scalar_t>(0.0f);
        }
      }
      __syncthreads();

      // looplevel 2: loop over KVtile blocks along seqLen dim
      //! For causal computation: kvTileIdx <= qTileIdx * gridDim.y +
      //! blockIdx.y
      // other working version: kvTileIdx < kvTileEnd (inefficient due to
      // loading of unnecessary numbers)
      const uint kvTileEnd = qTileIdx * gridDim.y + blockIdx.y + 1;
      for (uint kvTileIdx = 0; kvTileIdx < kvTileEnd; ++kvTileIdx) {

        //* offset in K&V matrix for kTile & vTile (global memory)
        // (k-tile & v-tile have the same BlockGlobalMemIdx, we just change the
        // pointer to the respective memory space to load the k-tile and
        // v-tile).
        const uint kvTileBlockGlobalMemIdx =
            batchHeadGridXGlobalMemIdxQKV + (dimHeads * KVtileDim) * kvTileIdx;

        //* (grid&block) offset X-axis in S = Q*K^T matrix
        // (along sequence dimension) (used for checking causality)
        const uint cTileGridXIdx = KVtileDim * kvTileIdx;
        const uint cTileBlockXIdx = cTileGridXIdx;

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
            //! while loading k: k = k / sqrt(dimHeads)
            SMEMARRAY(kTile, dimHeads, kvWarpTileThreadSharedMemYIdx,
                      kvWarpTileThreadSharedMemXIdx) =
                mul_g(matK[kvWarpTileThreadGlobalMemIdx],
                      float2type<scalar_t>(rsqrtf(type2float((dimHeads)))));
            SMEMARRAY(vTile, dimHeads, kvWarpTileThreadSharedMemYIdx,
                      kvWarpTileThreadSharedMemXIdx) =
                matV[kvWarpTileThreadGlobalMemIdx];
          }
        }
        __syncthreads();

        //! construct fTileCol for dTile computation (do only of kvTileIdx=0)
        // TODO maybe use a parallel scan for optimization (each thread
        // basically does the same computation)
        // fTileCol is the first column of
        // the fgates for the current dTile use all threads along Y (seqLen,
        // qTileDim) dimension to compute the sums of the forgetgates in
        // parallel

        // end idx for the chunkwise loop over fGatePreacts with flattened
        // thread block
        const uint fWarpChunkEnd = CEIL_DIV(QtileDim, blockDim.x * blockDim.y);
        if (kvTileIdx == 0) {
          // we begin at seqLen position 0 in X direction
          // we compute the cumulative sum of the forget gates per row position
          // in the dTile
          gridGroup.sync();
          // load fTileColLast from previous iteration and add to f_acc
          float fTileColLastVal =
              fTileColLast[batchHeadGridXGlobalMemIdxFtileColLast];

          // loop chunkwise over the fGatePreacts up to the current qTile
          // position
          const uint fChunkEnd = gridDim.y * qTileIdx + blockIdx.y + 1;
          const uint fChunkStart = gridDim.y * qTileIdx;
          //   max(0, gridDim.y * (qTileIdx - 1) + blockIdx.y + 1);
          const uint fChunkAccIterEnd = gridDim.y; // blockIdx.y + 1;
          for (uint fChunkAccIterIdx = 0; fChunkAccIterIdx < fChunkAccIterEnd;
               ++fChunkAccIterIdx) {
#ifdef DEBUG_fcolval2
            if ((blockIdx.x == 0) && (blockIdx.y == 0) && flatThreadIdx == 0) {
              printf("IDX: cTileBlockY=%d, qTileIdx=%d, fChunkIdx=%d (<%d), "
                     "blockIdx.y=%d, "
                     "flatThreadIdx=%d, kvTileIdx=%d, InitFTileColVal=%f, "
                     "fTileColLast=%f\n",
                     cTileBlockYIdx, qTileIdx, fChunkIdx, fChunkEnd, blockIdx.y,
                     flatThreadIdx, kvTileIdx, SMEMVECTOR(fTileCol, 3),
                     SMEMVECTOR(fTileColLast, 0));
            }
#endif
            //? f idxes
            // load fChunk for fChunkIdx
            //* (grid&block) offset in f preactivations for fChunk (global
            // memory)
            const uint fChunkGridXYGlobalMemIdx =
                batchHeadGridXGlobalMemIdxIFgateNMchunk +
                (1 * QtileDim) * gridDim.y * qTileIdx;
            const uint fChunkBlockGlobalMemIdx =
                fChunkGridXYGlobalMemIdx + (1 * QtileDim) * fChunkAccIterIdx;

            //! loading fChunk into shared memory with threadblocks
            for (uint fWarpChunkIdx = 0; fWarpChunkIdx < fWarpChunkEnd;
                 ++fWarpChunkIdx) {
              //? f idxes
              //* shared memory:
              const uint fThreadSharedMemYIdx =
                  flatThreadIdx + blockDim.x * blockDim.y * fWarpChunkIdx;
              //* global memory:
              const uint fThreadGlobalMemIdx =
                  fChunkBlockGlobalMemIdx + fThreadSharedMemYIdx;

              if (fThreadSharedMemYIdx < QtileDim) {
                SMEMVECTOR(fChunk, fThreadSharedMemYIdx) =
                    logsigmoid_g(fGatePreact[fThreadGlobalMemIdx]);
                // without logsigmoid for debugging only:
                // SMEMVECTOR(fChunk, fThreadSharedMemYIdx) =
                //     fGatePreact[fThreadGlobalMemIdx];
              }
            }
            __syncthreads();

            //! Construct fTileCol (first column of D matrix in a tiled way) by
            //! summing up the fgates
            // the very first forgetgate index must be f_2
            for (uint fWarpChunkIdx = 0; fWarpChunkIdx < fWarpChunkEnd;
                 ++fWarpChunkIdx) {
              //? f idxes
              //* shared memory:
              const uint fThreadSharedMemYIdx =
                  flatThreadIdx + blockDim.x * blockDim.y * fWarpChunkIdx;

              //? d idxes
              //* (thread) offset Y-axis (seqLen, QTileDim) of dTile (global)
              const uint dTileThreadYIdx =
                  cTileBlockYIdx + fThreadSharedMemYIdx;

              float f_acc;
              if (fThreadSharedMemYIdx < QtileDim) {
                // init forget gate accumulator
                if (fChunkAccIterIdx == 0) {
                  f_acc = 0.0f;
                } else {
                  f_acc = SMEMVECTOR(fTileCol, fThreadSharedMemYIdx);
                } // TODO from here (FIXED): this is the problem we need to set
                  // f_acc to 0 at first iteration, but we modify the start
                  // index

                // start the sum at the second index (corresponds to f_2)
                uint startIdx = 0;
                if ((qTileIdx == 0) && (fChunkAccIterIdx == 0)) {
                  startIdx = 1;
                }
                for (uint i = startIdx; i < QtileDim; ++i) {
                  //? f idxes
                  // fSumIdx corresponds to the current fGatePreact index
                  // (starting from 0) i.e. for f_2: fSumIdx = 1, for f_3:
                  // fSumIdx = 2, ...
                  const uint fSumIdx = gridDim.y * qTileIdx * QtileDim +
                                       fChunkAccIterIdx * QtileDim + i;
                  if (fSumIdx > dTileThreadYIdx) {
                    break;
                  }
                  f_acc = add_g(f_acc, type2float(SMEMVECTOR(fChunk, i)));
#ifdef DEBUG7
                  if ((blockIdx.x == 0) && (blockIdx.y == 1) &&
                      (flatThreadIdx == 6)) {
                    printf(
                        "qTileIdx=%d, fChunkIdx=%d (<%d), blockIdx.y=%d, "
                        "flatThreadIdx=%d: "
                        "fSumIdx=%d, dTileThreadYIdx=%d, f_acc=%f, fg(%d)=%f\n",
                        qTileIdx, fChunkIdx, fChunkEnd, blockIdx.y,
                        flatThreadIdx, fSumIdx, dTileThreadYIdx, f_acc, i,
                        type2float(SMEMVECTOR(fChunk, i)));
                  }
#endif
                }
                SMEMVECTOR(fTileCol, fThreadSharedMemYIdx) = f_acc;

#ifdef DEBUG_fcolval3
                if ((blockIdx.x == 0) && (blockIdx.y == 0) &&
                    flatThreadIdx == 7) {
                  printf(
                      "BIdx.y=%d, FLTIdx=%d: qTileIdx=%d, fChunkAccIterIdx=%d "
                      "(<%d), fWCIdx=%d, f_acc=%f, fTileColLast=%f\n",
                      blockIdx.y, flatThreadIdx, qTileIdx, fChunkAccIterIdx,
                      fChunkEnd, fWarpChunkIdx, f_acc, fTileColLastVal);
                }
#endif
              }
            } // end for fWarpChunkIdx loop (accumulator)
            __syncthreads();
          } // end for fChunkAccIterIdx loop

          //! Add fTileColLast to fTileCol
          for (uint fWarpChunkIdx = 0; fWarpChunkIdx < fWarpChunkEnd;
               ++fWarpChunkIdx) {
            //? f idxes
            //* shared memory:
            const uint fThreadSharedMemYIdx =
                flatThreadIdx + blockDim.x * blockDim.y * fWarpChunkIdx;

            if (fThreadSharedMemYIdx < QtileDim) {
              float fTileCol_val = SMEMVECTOR(fTileCol, fThreadSharedMemYIdx);
              fTileCol_val = add_g(fTileCol_val, fTileColLastVal);
              SMEMVECTOR(fTileCol, fThreadSharedMemYIdx) = fTileCol_val;

              if ((fThreadSharedMemYIdx == QtileDim - 1) &&
                  blockIdx.y == gridDim.y - 1) {
                fTileColLast[batchHeadGridXGlobalMemIdxFtileColLast] =
                    fTileCol_val;
#ifdef DEBUG_fcolval1
                if ((blockIdx.x == 0) && (blockIdx.y <= 1)) {
                  printf("STR: BIdx.y=%d: qTileIdx=%d, fWCIdx=%d (<%d), "
                         "flatThreadIdx=%d: fT_acc_res=%f, fTileCol_val=%f, "
                         "fTileColLastVal=%f\n",
                         blockIdx.y, qTileIdx, fWarpChunkIdx, fWarpChunkEnd,
                         flatThreadIdx,
                         SMEMVECTOR(fTileCol, fThreadSharedMemYIdx),
                         fTileCol_val, fTileColLastVal);
                }
#endif
              }

            } // end if fThreadSharedMemYIdx < QtileDim
          }   // end for fWarpChunkIdx loop (addition)
        }     // end if kvTileIdx == 0

        // else: do nothing
        // we are within the sequence at position > kvTileDim * kvTileIdx
        // we can just use the fTileCol from the previous iteration and keep
        // subtracting we only need to update the fTileCol for the next
        // kvTileIdx at the end of the current kvTileIdx iteration

#ifdef DEBUG8
        if ((blockIdx.x == 0) && (blockIdx.y == 0) && (flatThreadIdx == 0)) {
          printf("qTileIdx=%d, kvTileIdx=%d, cTileBlockXIdx=%d, "
                 "cTileBlockYIdx=%d\n",
                 qTileIdx, kvTileIdx, cTileBlockXIdx, cTileBlockYIdx);
        }
#endif
        //! iChunk&fChunk Loading (only once per kvTileIdx in KVtileDim)
        //* (grid&block) offset in i&f preactivations for i and f chunk
        // every thread block loads the same i&f preactivations
        const uint ifChunkBlockXYGlobalMemIdx =
            batchHeadGridXGlobalMemIdxIFgateNMchunk +
            (1 * KVtileDim) * kvTileIdx;

        // Y: seqLen (or KVtileDim), X: 1
        // we only load the fGatePreacts for the current kvTileIdx
        const uint ifChunkChunkYEnd =
            CEIL_DIV(KVtileDim, blockDim.x * blockDim.y);
        for (uint iChunkYIdx = 0; iChunkYIdx < ifChunkChunkYEnd; ++iChunkYIdx) {
          //? i idxes
          //* shared memory:
          const uint ifThreadSharedMemYIdx =
              flatThreadIdx + blockDim.x * blockDim.y * iChunkYIdx;
          //* global memory:
          const uint ifChunkThreadGlobalMemIdx =
              ifChunkBlockXYGlobalMemIdx + ifThreadSharedMemYIdx;

          if (ifThreadSharedMemYIdx < KVtileDim) {
            SMEMVECTOR(iChunk, ifThreadSharedMemYIdx) =
                iGatePreact[ifChunkThreadGlobalMemIdx];
            SMEMVECTOR(fChunk, ifThreadSharedMemYIdx) =
                logsigmoid_g(fGatePreact[ifChunkThreadGlobalMemIdx]);
            // without logsigmoid for debugging only:
            // SMEMVECTOR(fChunk, ifThreadSharedMemYIdx) =
            //     fGatePreact[ifChunkThreadGlobalMemIdx];
          }
        }
        __syncthreads();

        //! construct dTile
        // go over tile from left to right in kvTileDim dimension, subtract
        // fgates again, add igates, and - while going over it - compute the
        // max state for the dTile (keep max in register and copy to shared
        // memory at the end)

        for (uint fWarpChunkIdx = 0; fWarpChunkIdx < fWarpChunkEnd;
             ++fWarpChunkIdx) {
          //? f idxes
          //* shared memory:
          const uint fThreadSharedMemYIdx =
              flatThreadIdx + blockDim.x * blockDim.y * fWarpChunkIdx;

          //? d idxes
          //* (thread) [local] offset Y-axis (seqLen, QTileDim) of dTile
          const uint dTileLocalThreadYIdx = fThreadSharedMemYIdx;
          //* (thread) [global] offset Y-axis (seqLen, QTileDim) of dTile
          const uint dTileThreadYIdx = cTileBlockYIdx + dTileLocalThreadYIdx;

          if (fThreadSharedMemYIdx < QtileDim) {

            float f_acc_subtractfrom =
                SMEMVECTOR(fTileCol, fThreadSharedMemYIdx);
            float d_max = -CUDART_INF_F; // init to -inf, so first max is
                                         // always larger than -inf
            float d_val = 0.0f;
            for (uint i = 0; i < KVtileDim; ++i) {
              //* (thread) [global] offset X-axis (KVtileDim) of dTile
              const uint dTileThreadXIdx = cTileBlockXIdx + i;

              // f gate only
              if (dTileThreadXIdx == dTileThreadYIdx) {
                // set to 0
                f_acc_subtractfrom = 0.0f;
              } else if (dTileThreadXIdx == 0) {
                // first column of dTile
                // no change to f_acc_subtractfrom
              } else if (dTileThreadXIdx > dTileThreadYIdx) {
                // set to negative infinity
                f_acc_subtractfrom = -CUDART_INF_F;
              } else {
                // dTileThreadXIdx < dTileThreadYIdx
                // subtract f gate
                f_acc_subtractfrom = sub_g(f_acc_subtractfrom,
                                           type2float(SMEMVECTOR(fChunk, i)));
              }
              // Debugging only: f_gate only:
              //   d_val = f_acc_subtractfrom;

              d_val =
                  add_g(f_acc_subtractfrom, type2float(SMEMVECTOR(iChunk, i)));

              // max state
              d_max = max_g(d_max, d_val);

              // write d_val into dTile shared memory
              SMEMARRAY(dTile, KVtileDim, dTileLocalThreadYIdx, i) =
                  float2type<scalar_t>(d_val);
#ifdef DEBUG9
              if ((blockIdx.x == 0) && (blockIdx.y == 0) &&
                  (flatThreadIdx == 7)) {
                printf("qTileIdx=%d, kvTileIdx=%d, "
                       "cTileBlockXIdx=%d, "
                       "cTileBlockYIdx=%d, "
                       "dTileThreadXYIdx=(%d,%d), "
                       "d_val=%f\n",
                       qTileIdx, kvTileIdx, cTileBlockXIdx, cTileBlockYIdx,
                       dTileThreadXIdx, dTileThreadYIdx, d_val);
              }
#endif
            }
            // save max state of dTile in shared memory
            SMEMVECTOR(mChunk, dTileLocalThreadYIdx) =
                float2type<scalar_t>(d_max);
            // save last f_acc_subtractfrom in fTileCol for next kvTileIdx
            SMEMVECTOR(fTileCol, fThreadSharedMemYIdx) = f_acc_subtractfrom;
          }
        }
        __syncthreads();

#ifdef OUTPUT_matD
        //! DEBUG only: write dTile to global memory
        // left upper corner of cWarpTileBlock in C (global memory)
        //* cdTile Global Memory Index (Debug only)
        const uint cdTileGridXYGlobalMemIdx =
            batchHeadGridXGlobalMemIdxCD +
            (seqLen * QtileDim * gridDim.y) * qTileIdx;
        const uint cdTileBlockGlobalMemIdx = cdTileGridXYGlobalMemIdx +
                                             (seqLen * QtileDim) * blockIdx.y +
                                             (kvTileIdx * KVtileDim);

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
                SMEMARRAY(dTile, KVtileDim, cdWarpTileThreadSharedMemYIdx,
                          cdWarpTileThreadSharedMemXIdx);
#ifdef DEBUG10
            if ((blockIdx.x == 0) && (blockIdx.y == 0) &&
                (threadIdx.x == 0 && threadIdx.y == 0)) {
              printf("qTileIdx=%d, kvTileIdx=%d, cTileBlockXIdx=%d, "
                     "cTileBlockYIdx=%d, dTileThreadXYIdx=(%d,%d), "
                     "d_val=%f\n",
                     qTileIdx, kvTileIdx, cTileBlockXIdx, cTileBlockYIdx,
                     cdWarpTileThreadSharedMemXIdx,
                     cdWarpTileThreadSharedMemYIdx,
                     type2float(SMEMARRAY(dTile, KVtileDim,
                                          cdWarpTileThreadSharedMemYIdx,
                                          cdWarpTileThreadSharedMemXIdx)));
            }
#endif
          }
        }
#endif

        //! compute S = (Q x K^T)
        // (QtileDim,KVtileDim) = (QtileDim,dimHeads) x (dimHeads,KVtileDim)
        // loops over cTile rows (outer) and columns (inner)
        const uint cWarpTileYEnd = CEIL_DIV(QtileDim, blockDim.y);
        const uint cWarpTileXEnd = CEIL_DIV(KVtileDim, blockDim.x);
        for (uint cWarpTileYIdx = 0; cWarpTileYIdx < cWarpTileYEnd;
             ++cWarpTileYIdx) {
          //* (thread) offset Y-axis in S = Q*K^T
          const uint cTileThreadYIdx =
              cTileBlockYIdx + blockDim.y * cWarpTileYIdx + threadIdx.y;

          for (uint cWarpTileXIdx = 0; cWarpTileXIdx < cWarpTileXEnd;
               ++cWarpTileXIdx) {
            //? cTileIdxes
            //* shared memory:
            const uint cWarpTileThreadSharedMemYIdx =
                blockDim.y * cWarpTileYIdx + threadIdx.y;
            const uint cWarpTileThreadSharedMemXIdx =
                blockDim.x * cWarpTileXIdx + threadIdx.x;

            //* (thread) offset X-axis in S = Q*K^T
            const uint cTileThreadXIdx =
                cTileBlockXIdx + blockDim.x * cWarpTileXIdx + threadIdx.x;

            // scalar_t qk_acc = dscalar_zero<scalar_t>();
            float qk_acc = 0.0f;
            //! check for causality here
            // compute only the lower triangle (below main diagonal) of S =
            // Q*K^T
            if (cTileThreadXIdx <= cTileThreadYIdx) {
              for (uint i = 0; i < dimHeads; ++i) {
                qk_acc = add_g(
                    qk_acc, type2float(mul_g(
                                SMEMARRAY(qTile, dimHeads,
                                          cWarpTileThreadSharedMemYIdx, i),
                                SMEMARRAY(kTile, dimHeads,
                                          cWarpTileThreadSharedMemXIdx, i))));
#ifdef DEBUG4
                if ((blockIdx.x == 0) && (blockIdx.y == 0) &&
                    (threadIdx.x == 0) && (threadIdx.y == 3) &&
                    (cWarpTileXIdx == 0) && (kvTileIdx == 0) &&
                    (i == dimHeads - 1)) {
                  printf("qTIdx=%d|kvTIdx=%d: qTile[%d][%d] = "
                         "%f\n",
                         qTileIdx, kvTileIdx, cWarpTileThreadSharedMemYIdx, i,
                         type2float(qTile[cWarpTileThreadSharedMemYIdx][i]));
                  printf("qTIdx=%d|kvTIdx=%d: kTile[%d][%d] = "
                         "%f\n",
                         qTileIdx, kvTileIdx, cWarpTileThreadSharedMemXIdx, i,
                         type2float(kTile[cWarpTileThreadSharedMemXIdx][i]));
                  printf("qTIdx=%d|kvTIdx=%d: "
                         "cTile[%d][%d](%d) = %f\n",
                         qTileIdx, kvTileIdx, cWarpTileThreadSharedMemYIdx,
                         cWarpTileThreadSharedMemXIdx, i, type2float(qk_acc));
                }
#endif
              }
            }
            SMEMARRAY(cTile, KVtileDim, cWarpTileThreadSharedMemYIdx,
                      cWarpTileThreadSharedMemXIdx) =
                float2type<scalar_t>(qk_acc);
            __syncthreads();
          }
        }

        //! compute C_tilde: multiply S with dTile, i.e. fill cTile
        //! compute "raw normalizer" l: rowsum of cTile
        //! compute normalizer n: max(abs(l),exp(-m))
        // use flattened threads for the rowsum, i.e. each thread computes the
        // sum of a row of cTile
        // l and n are chunks (of 1D vectors) of size (QtileDim x 1)
        const uint lWarpChunkEnd = CEIL_DIV(QtileDim, blockDim.x * blockDim.y);
        for (uint lWarpChunkIdx = 0; lWarpChunkIdx < lWarpChunkEnd;
             ++lWarpChunkIdx) {

          //? l idxes
          //* shared memory:
          const uint lThreadSharedMemYIdx =
              flatThreadIdx + blockDim.x * blockDim.y * lWarpChunkIdx;

          if (lThreadSharedMemYIdx < QtileDim) {
            // compute m_val
            scalar_t m_prev_val = SMEMVECTOR(mPrevChunk, lThreadSharedMemYIdx);
            scalar_t m_val =
                max_g(m_prev_val, SMEMVECTOR(mChunk, lThreadSharedMemYIdx));
            SMEMVECTOR(mChunk, lThreadSharedMemYIdx) = m_val;

            // compute c_tilde_val in lower triangle of C Matrix
            //? c idxes
            const uint cTileGlobalThreadYIdx =
                cTileBlockYIdx + lThreadSharedMemYIdx;
            float l_acc = 0.0f;
            for (uint i = 0; i < KVtileDim; ++i) {
              const uint cTileGlobalThreadXIdx = cTileBlockXIdx + i;
              if (cTileGlobalThreadXIdx > cTileGlobalThreadYIdx) {
                // values above the main diagonal in D' are
                // 0
                SMEMARRAY(dTile, KVtileDim, lThreadSharedMemYIdx, i) =
                    dscalar_zero<scalar_t>();
              } else {
                scalar_t s_val =
                    SMEMARRAY(cTile, KVtileDim, lThreadSharedMemYIdx, i);
                scalar_t d_val =
                    SMEMARRAY(dTile, KVtileDim, lThreadSharedMemYIdx, i);

                // store c_tilde_val in dTile for now (later
                // in cTile) for debugging only (since dTile
                // is already written to
                //   global memory)
                scalar_t c_tilde_val = mul_g(s_val, exp_g(sub_g(d_val, m_val)));
                SMEMARRAY(dTile, KVtileDim, lThreadSharedMemYIdx, i) =
                    c_tilde_val;

                l_acc = add_g(l_acc, type2float(c_tilde_val));
              }
            }
            // l_acc is the rowsum of cTile
            // compute l_val = exp(m_prev - m) * l_prev + l_acc
            scalar_t l_prev_val = SMEMVECTOR(lPrevChunk, lThreadSharedMemYIdx);
            scalar_t l_val =
                add_g(mul_g(exp_g(sub_g(m_prev_val, m_val)), l_prev_val),
                      float2type<scalar_t>(l_acc));
            SMEMVECTOR(lChunk, lThreadSharedMemYIdx) = l_val;

            // compute n_val
            scalar_t n_val = max_g(abs_g(l_val), exp_g(neg_g(m_val)));
            SMEMVECTOR(nChunk, lThreadSharedMemYIdx) = n_val;
#ifdef DEBUG12
            if ((blockIdx.x == 0) && (blockIdx.y == 0)) {
              printf("qTileIdx=%d, kvTileIdx=%d, cTBlXIdx=%d, "
                     "cTBlYIdx=%d, lThreadSMIdx=%d, tbIdxXY=(%d,%d): "
                     "l_val=%f, l_prev_val=%f, l_acc=%f, exp(-m_val)=%f, "
                     "n_val=%f\n",
                     qTileIdx, kvTileIdx, cTileBlockXIdx, cTileBlockYIdx,
                     lThreadSharedMemYIdx, threadIdx.x, threadIdx.y,
                     type2float(l_val), type2float(l_prev_val), l_acc,
                     type2float(exp_g(neg_g(m_val))), type2float(n_val));
            }
#endif
          }
        } // end: lWarpChunkIdx
        __syncthreads();

        //! compute H += C * V, i.e. fill hTile
        //! accumulate KVtiles to hTile
        // (QtileDim,dimHeads) = (QtileDim,KVtileDim) x (KVtileDim,dimHeads)
        // loops over hTile rows (outer) and columns (inner)
        const uint hWarpTileYEnd = CEIL_DIV(QtileDim, blockDim.y);
        const uint hWarpTileXEnd = CEIL_DIV(dimHeads, blockDim.x);
        for (uint hWarpTileYIdx = 0; hWarpTileYIdx < hWarpTileYEnd;
             ++hWarpTileYIdx) {
          //? hTileIdxes
          //* shared memory:
          const uint hWarpTileThreadSharedMemYIdx =
              blockDim.y * hWarpTileYIdx + threadIdx.y;

          scalar_t n_val = SMEMVECTOR(nChunk, hWarpTileThreadSharedMemYIdx);
          scalar_t n_prev_val =
              SMEMVECTOR(nPrevChunk, hWarpTileThreadSharedMemYIdx);
          scalar_t m_val = SMEMVECTOR(mChunk, hWarpTileThreadSharedMemYIdx);
          scalar_t m_prev_val =
              SMEMVECTOR(mPrevChunk, hWarpTileThreadSharedMemYIdx);

#ifdef DEBUG11
          if ((blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0)) {
            printf("qTileIdx=%d, kvTileIdx=%d, cTileBlockXIdx=%d, "
                   "cTileBlockYIdx=%d, hWTileSMIdxY=%d, tbIdxXY=(%d,%d): "
                   "m_val=%f, m_prev_val=%f, n_val=%f, n_prev_val=%f\n",
                   qTileIdx, kvTileIdx, cTileBlockXIdx, cTileBlockYIdx,
                   hWarpTileThreadSharedMemYIdx, threadIdx.x, threadIdx.y,
                   type2float(m_val), type2float(m_prev_val), type2float(n_val),
                   type2float(n_prev_val));
          }
#endif

          // weight = exp(m_prev - m) * n_prev / n
          scalar_t weighting_factor_h_prev =
              mul_g(exp_g(sub_g(m_prev_val, m_val)), div_g(n_prev_val, n_val));

          for (uint hWarpTileXIdx = 0; hWarpTileXIdx < hWarpTileXEnd;
               ++hWarpTileXIdx) {

            //? hTileIdxes
            //* shared memory:
            const uint hWarpTileThreadSharedMemXIdx =
                blockDim.x * hWarpTileXIdx + threadIdx.x;

            float sv_acc = 0.0f;
            for (uint i = 0; i < KVtileDim; ++i) {
              // compute c_val: c_val = c_tilde_val / n_val
              scalar_t c_tilde_val =
                  SMEMARRAY(dTile, KVtileDim, hWarpTileThreadSharedMemYIdx, i);
              scalar_t c_val = div_g(c_tilde_val, n_val);

              sv_acc = add_g(
                  sv_acc, type2float(mul_g(
                              c_val, SMEMARRAY(vTile, dimHeads, i,
                                               hWarpTileThreadSharedMemXIdx))));
            }

            // accumulate over all KVtiles
            if (kvTileIdx == 0) {
              // we need to clear the hTile in first iteration
              SMEMARRAY(hTile, dimHeads, hWarpTileThreadSharedMemYIdx,
                        hWarpTileThreadSharedMemXIdx) =
                  float2type<scalar_t>(sv_acc);
            } else {
              scalar_t h_prev_val =
                  SMEMARRAY(hTile, dimHeads, hWarpTileThreadSharedMemYIdx,
                            hWarpTileThreadSharedMemXIdx);
              // reweight the previous value
              scalar_t weighted_h_prev_val =
                  mul_g(weighting_factor_h_prev, h_prev_val);
              // formulas
              SMEMARRAY(hTile, dimHeads, hWarpTileThreadSharedMemYIdx,
                        hWarpTileThreadSharedMemXIdx) =
                  add_g(weighted_h_prev_val, float2type<scalar_t>(sv_acc));
            }
            __syncthreads();
          }
        }

        //! move to next kvTileIdx
        // update lPrevChunk, mPrevChunk, nPrevChunk
        for (uint lWarpChunkIdx = 0; lWarpChunkIdx < lWarpChunkEnd;
             ++lWarpChunkIdx) {
          //? l idxes
          //* shared memory:
          const uint lThreadSharedMemYIdx =
              flatThreadIdx + blockDim.x * blockDim.y * lWarpChunkIdx;

          if (lThreadSharedMemYIdx < QtileDim) {
            SMEMVECTOR(lPrevChunk, lThreadSharedMemYIdx) =
                SMEMVECTOR(lChunk, lThreadSharedMemYIdx);
            SMEMVECTOR(mPrevChunk, lThreadSharedMemYIdx) =
                SMEMVECTOR(mChunk, lThreadSharedMemYIdx);
            SMEMVECTOR(nPrevChunk, lThreadSharedMemYIdx) =
                SMEMVECTOR(nChunk, lThreadSharedMemYIdx);
          }
        }
      } // end looplevel 2: kvTileIdx

      // TODO sync all blocks here, necessary? The loop above has different
      // number of iterations for each block
      // gridGroup.sync();

      //! write hTile to global memory (has the same memory index as qTile)
      // loops over hTile rows (outer) and columns (inner)
      const uint hWarpTileYEnd = CEIL_DIV(QtileDim, blockDim.y);
      const uint hWarpTileXEnd = CEIL_DIV(dimHeads, blockDim.x);
      for (uint hWarpTileYIdx = 0; hWarpTileYIdx < hWarpTileYEnd;
           ++hWarpTileYIdx) {
        for (uint hWarpTileXIdx = 0; hWarpTileXIdx < hWarpTileXEnd;
             ++hWarpTileXIdx) {

          //? cTileIdxes
          //* shared memory:
          const uint hWarpTileThreadSharedMemYIdx =
              blockDim.y * hWarpTileYIdx + threadIdx.y;
          const uint hWarpTileThreadSharedMemXIdx =
              blockDim.x * hWarpTileXIdx + threadIdx.x;
          //* global memory:
          // left upper corner of cWarpTileBlock in C (global memory)
          const uint hWarpTileBlockGlobalMemIdx =
              qTileBlockGlobalMemIdx + (dimHeads * blockDim.y) * hWarpTileYIdx +
              blockDim.x * hWarpTileXIdx;
          const uint hWarpTileThreadGlobalMemIdx =
              hWarpTileBlockGlobalMemIdx + dimHeads * threadIdx.y + threadIdx.x;

          matH[hWarpTileThreadGlobalMemIdx] =
              SMEMARRAY(hTile, dimHeads, hWarpTileThreadSharedMemYIdx,
                        hWarpTileThreadSharedMemXIdx);
        }
      }
      __syncthreads();

      //* global memory:
      const uint nmChunkGridXYGlobalMemIdx =
          batchHeadGridXGlobalMemIdxIFgateNMchunk +
          (1 * QtileDim * gridDim.y) * qTileIdx;
      const uint nmChunkBlockGlobalMemIdx =
          nmChunkGridXYGlobalMemIdx + (1 * QtileDim) * blockIdx.y;

      //! write nChunk and mChunk to global memory for backward pass
      const uint nmChunkYEnd = CEIL_DIV(QtileDim, blockDim.x * blockDim.y);
      for (uint nmChunkYIdx = 0; nmChunkYIdx < nmChunkYEnd; ++nmChunkYIdx) {
        //? n&m idxes
        //* shared memory:
        const uint nmThreadSharedMemYIdx =
            flatThreadIdx + blockDim.x * blockDim.y * nmChunkYIdx;
        //* global memory:
        const uint nmThreadGlobalMemIdx =
            nmChunkBlockGlobalMemIdx + nmThreadSharedMemYIdx;

        if (nmThreadSharedMemYIdx < QtileDim) {
          // write nChunk and mChunk to global memory
          vecN[nmThreadGlobalMemIdx] =
              SMEMVECTOR(nChunk, nmThreadSharedMemYIdx);
          vecM[nmThreadGlobalMemIdx] =
              SMEMVECTOR(mChunk, nmThreadSharedMemYIdx);

          // optionally write lChunk to global memory and compute nChunk from
          // mChunk and lChunk
        }
      }
    } // end looplevel 1: qTileIdx
    __syncthreads();
  } // end looplevel 0: batchHeadIdx
} // end vlstm_fw() kernel

template <typename scalar_t>
void kernel_dispatchers::vlstm_fw_dispatch(
    scalar_t *matH, scalar_t *vecN, scalar_t *vecM, scalar_t *matC,
    scalar_t *matQ, scalar_t *matK, scalar_t *matV, scalar_t *iGatePreact,
    scalar_t *fGatePreact, int batchSize, int numHeads, int seqLen,
    int dimHeads) {
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
  // we are storing the following tiles in shared memory:
  // - Input tiles: qTile -> (QtileDim, dimHeads +
  // SHARED_MEM_PADDING), vTile, kTile -> (KVtileDim, dimHeads +
  // SHARED_MEM_PADDING)
  // TODO from here add input & forgetgate tiles
  // - Intermediate result tile: cTile, dTile -> (QtileDim, KVtileDim +
  // SHARED_MEM_PADDING)
  // - Output tile: hTile -> (QtileDim, dimHeads + SHARED_MEM_PADDING)

  const uint qhTileSharedMemSize =
      sizeof(scalar_t) * QtileDim * (dimHeads + SHARED_MEM_PADDING);
  const uint kvTileSharedMemSize =
      sizeof(scalar_t) * KVtileDim * (dimHeads + SHARED_MEM_PADDING);
  const uint cdTileSharedMemSize =
      sizeof(scalar_t) * QtileDim * (KVtileDim + SHARED_MEM_PADDING);

  // See here:
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses
  // the idea of the padding is that every number is stored in a different
  // memory bank this should help to avoid bank conflicts as many threads need
  // to access the same input and forget gate values at the same time for the
  // gate matrix computation
  // TODO check if this is really helping!
  const uint iChunkSharedMemSize =
      sizeof(scalar_t) * KVtileDim * (1 + SHARED_MEM_PADDING);
  const uint fChunkSharedMemSize =
      sizeof(scalar_t) * QtileDim * (1 + SHARED_MEM_PADDING);

  // we keep these as float as it acts as accumulator
  const uint fTileColSharedMemSize =
      sizeof(float) * QtileDim * (1 + SHARED_MEM_PADDING);

  const uint mChunkSharedMemSize =
      sizeof(scalar_t) * QtileDim * (1 + SHARED_MEM_PADDING);
  const uint lChunkSharedMemSize =
      sizeof(scalar_t) * QtileDim * (1 + SHARED_MEM_PADDING);
  const uint nChunkSharedMemSize =
      sizeof(scalar_t) * QtileDim * (1 + SHARED_MEM_PADDING);

  // Input/Output tiles: 4x for qTile, vTile, kTile, hTile
  // Intermediate tiles: 2x for cTile, dTile
  // Intermediate tiles: 2x for mChunk, lChunk
  const uint sharedMemorySize =
      2 * qhTileSharedMemSize + 2 * kvTileSharedMemSize +
      2 * cdTileSharedMemSize + iChunkSharedMemSize + fChunkSharedMemSize +
      fTileColSharedMemSize + 2 * mChunkSharedMemSize +
      2 * lChunkSharedMemSize + 2 * nChunkSharedMemSize;

  printf("blocksxy: %d-%d, threadsxy: %d-%d, shared_mem in bytes: %d\n",
         gridDims.x, gridDims.y, blockDims.x, blockDims.y, sharedMemorySize);
  // cudaSetDevice(0);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  //* allocate intermediate HBM memory for D matrix (forget gate
  // preactivations)
  const uint fTileColLastGlobalMemSize = sizeof(float) * batchSize * numHeads;
  float *fTileColLast;
  gpuErrchk(cudaMalloc((void **)&fTileColLast, fTileColLastGlobalMemSize));
  gpuErrchk(cudaMemset(fTileColLast, 0, fTileColLastGlobalMemSize));

  auto kernel = kernels::vlstm_fw<scalar_t, TblockDim, QtileDim, KVtileDim>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       sharedMemorySize);
  // define void* pointers to the kernel arguments
  void *kernelArgs[] = {
      (void *)&matH,         (void *)&vecN,        (void *)&vecM,
      (void *)&matC,         (void *)&matQ,        (void *)&matK,
      (void *)&matV,         (void *)&iGatePreact, (void *)&fGatePreact,
      (void *)&fTileColLast, (void *)&batchSize,   (void *)&numHeads,
      (void *)&seqLen,       (void *)&dimHeads};

  cudaLaunchCooperativeKernel((void *)kernel, gridDims, blockDims, kernelArgs,
                              sharedMemorySize, stream);

  gpuErrchk(cudaPeekAtLastError());

  // free the allocated memory
  gpuErrchk(cudaFree(fTileColLast));

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  gpuErrchk(cudaDeviceSynchronize());

  // gpuErrchk(cudaPeekAtLastError());
  // gpuErrchk(cudaDeviceSynchronize());
}

// this is needed to make sure that the compiler instantiates the template
template void kernel_dispatchers::vlstm_fw_dispatch<__nv_bfloat16>(
    __nv_bfloat16 *matH, __nv_bfloat16 *vecN, __nv_bfloat16 *vecM,
    __nv_bfloat16 *matC, __nv_bfloat16 *matQ, __nv_bfloat16 *matK,
    __nv_bfloat16 *matV, __nv_bfloat16 *iGatePreact, __nv_bfloat16 *fGatePreact,
    int batchSize, int numHeads, int seqLen, int dimHeads);
template void kernel_dispatchers::vlstm_fw_dispatch<__half>(
    __half *matH, __half *vecN, __half *vecM, __half *matC, __half *matQ,
    __half *matK, __half *matV, __half *iGatePreact, __half *fGatePreact,
    int batchSize, int numHeads, int seqLen, int dimHeads);
template void kernel_dispatchers::vlstm_fw_dispatch<float>(
    float *matH, float *vecN, float *vecM, float *matC, float *matQ,
    float *matK, float *matV, float *iGatePreact, float *fGatePreact,
    int batchSize, int numHeads, int seqLen, int dimHeads);

} // namespace vlstm