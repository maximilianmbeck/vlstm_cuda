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
#include <mma.h>

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

namespace nv = nvcuda;

namespace kernels {

template <typename scalar_t, int QblockDim, int KVblockDim>
__global__ void vlstm_fw(scalar_t *matH, scalar_t *vecN, scalar_t *vecM,
                         scalar_t *matC, scalar_t *matQ, scalar_t *matK,
                         scalar_t *matV, scalar_t *iGatePreact,
                         scalar_t *fGatePreact, float *fTileColLast,
                         int batchSize, int numHeads, int seqLen, int dimHeads);

} // namespace kernels

////////////////////////////////////////////////////////////////////////////////////////
// GPU SPECIFIC
#define WARP_SIZE 32
// combine 4 warps to a group
#define WARP_GROUP_SIZE 4
//

// Tensor Core MxN = QTCxKVC dimensions (K is always 16)
#define QTC_DIM 16  // M
#define KVTC_DIM 16 // N
#define KTC_DIM 16  // K
//

// SHMEM TILE DIMENSIONS
#define KVTILE_DIM 64 // KVtileDim: TileDim for K&V along seqLen dim
// QTILE_DIM must be divisible by KVTILE_DIM and TBLOCK_DIM,
// KVTILE_DIM <= QTILE_DIM
#define QTILE_DIM 64 // QtileDim: TileDim for Q along seqLen dim
//

// WARP SETUP
#define NUM_WARP_GROUPS (QTILE_DIM / (WARP_GROUP_SIZE * QTC_DIM))
#define NUM_WARPS (NUM_WARP_GROUPS * WARP_GROUP_SIZE)
//

// MM (Matrix Multiply) SETUP
#define NUM_KDIM_FRAGMENTS_A 1
#define NUM_KDIM_FRAGMENTS_B 1
//

// we use the float4 load/store instructions to load 4 floats at once
// we choose the layout such that one warp can load a 8x32 x 2byte tile
#define GMEM_LOAD_BLOCK_2BITEMS_X 32
// 4 warps load a 32x32 x 2byte tile
// 8 warps load a 64x32 x 2byte tile
// TODO make float4 load/store instructions work -> then set to 8
#define GMEM_LOAD_THREAD_2BITEMS_X                                             \
  8 // 1 // float4 = 16 bytes / 2 bytes per element
// number of threads in a row to load from global memory:
#define GMEM_LOAD_WARP_COLS_X (WARP_SIZE / GMEM_LOAD_THREAD_2BITEMS_X)
#define GMEM_LOAD_WARP_ROWS_Y (WARP_SIZE / GMEM_LOAD_WARP_COLS_X)

// number of thread columns that all threads / warps load from global memory
#define GMEM_LOAD_BLOCK_COLS_X (GMEM_LOAD_WARP_COLS_X)
// number of thread rows that all threads / warps load from global memory
#define GMEM_LOAD_BLOCK_ROWS_Y (GMEM_LOAD_WARP_ROWS_Y * NUM_WARPS)

// SHARED MEM SETUP & PADDING
// shared memory must be aligned: depends on scalar_t (multiples of 4 should be
// fine for bf16, fp16 and fp32)
// shared memory padding for 2D tiles / arrays in shared memory
//! In order for float4/int4 accesses to be aligned, padding must be 16 bytes
// otherwise kernel will crash
#define SHARED_MEM_PADDING_TILE 16
// shared memory padding for 1D vectors in shared memory
// TODO find correct padding here
#define SHARED_MEM_PADDING_CHUNK 4
//

// SMEMARRAY: access shared memory array (2D)
#define SMEMARRAY(array, stride, row, col)                                     \
  array[(row) * (stride + SHARED_MEM_PADDING_TILE) + (col)]
// SMEMVECTOR: access shared memory vector (1D)
#define SMEMVECTOR(array, idx) array[(idx) * (1 + SHARED_MEM_PADDING_CHUNK)]

#define DEBUG 1
// #define DEBUG2 1
// #define DEBUG4 1
// #define DEBUG5 1
// #define DEBUG8 1
// #define DEBUG9 1
// #define DEBUG10 1
// #define DEBUG11 1
// #define DEBUG12 1

// #define DEBUG_GMEMSTREAM1 1
// #define DEBUG_GMEMSTREAM2 1
// #define DEBUG_GMEMSTREAM3 1
// #define DEBUG_GMEMSTREAM4 1
// #define DEBUG_GMEMSTREAM5 1
// #define DEBUG_GMEMSTREAM6 1

// #define DEBUG_fcolval1 1
// #define DEBUG_fcolval2 1
// #define DEBUG_fcolval3 1
// #define DEBUG_fcolval4 1
// #define DEBUG_fcolval5 1

// #define DEBUG_gridSize1 1

// #define DEBUG_hsout1 1
// #define DEBUG_hsout2 1

// #define OUTPUT_matD 1
#define OUTPUT_matS 1

#define COMPUTE_QK_TENSORCORE 1

// #define INCL_DMAT_COMP 1

/**
Conventions:
- chunk: A 1D vector in shared memory
- tile: A 2D matrix in shared memory
*/

/* vLSTM Forward Kernel v0 */

template <typename scalar_t, int QtileDim, int KVtileDim>
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
    printf("In FW-Kernel: QtileDim: %d, KVtileDim: %d\n", QtileDim, KVtileDim);
  }
#endif
  cg::grid_group gridGroup = cg::this_grid();

  //! Shared Memory aka SRAM
  // the data in this shared memory is shared across all threads in a thread
  // block
  extern __shared__ float sbuf[]; // declare it as float and redefine it later
                                  // TODO from here
  //? for inputs
  // Note: keep in mind the memory is defined in a contiguous region.
  // One pointer has the full memory space until the next point is defined.
  // Therefore to read the size of a single shared memory array you need to
  // have a look at the offset for the next array.

  // qtile (QtileDim x dimHeads) in shared memory (padding for alignment)
  scalar_t *qTile = (scalar_t *)sbuf;
  // kTile and vTile (KVtileDim x dimHeads) in shared memory
  scalar_t *kvTile =
      (scalar_t *)&qTile[QtileDim * (dimHeads + SHARED_MEM_PADDING_TILE)];

  //? for intermediate results
  // init cTile (QtileDim x KVTileDim) in shared memory for intermediate
  // result of QK^T
  // TODO initialize cTile to 0
  scalar_t *cTile =
      (scalar_t *)&kvTile[KVtileDim * (dimHeads + SHARED_MEM_PADDING_TILE)];
  // init result hTile (QTileDim x dimHeads) in shared memory
  scalar_t *hTile =
      (scalar_t *)&cTile[QtileDim * (KVtileDim + SHARED_MEM_PADDING_TILE)];
  // init dTile (QTileDim x KVTileDim) in shared memory for forget and input
  // gate matrix
  scalar_t *dTile =
      (scalar_t *)&hTile[QtileDim * (dimHeads + SHARED_MEM_PADDING_TILE)];

  //? for input and forget gate
  // init iChunk (KVTileDim x 1) in shared memory for input gate
  scalar_t *iChunk =
      (scalar_t *)&dTile[QtileDim * (dimHeads + SHARED_MEM_PADDING_TILE)];
  // init fChunk (QTileDim x 1) in shared memory for forget gate
  scalar_t *fChunk =
      (scalar_t *)&iChunk[KVtileDim * (1 + SHARED_MEM_PADDING_CHUNK)];

  // init mChunk (QTileDim x 1) in shared memory for max state of
  // dTile
  scalar_t *mChunk =
      (scalar_t *)&fChunk[QtileDim * (1 + SHARED_MEM_PADDING_CHUNK)];
  // init mPrevTileCol (QTileDim x 1) in shared memory for previous
  // max state of dTile
  scalar_t *mPrevChunk =
      (scalar_t *)&mChunk[QtileDim * (1 + SHARED_MEM_PADDING_CHUNK)];
  // init lChunk (QTileDim x 1) in shared memory for rowsum of cTile * dTile
  scalar_t *lChunk =
      (scalar_t *)&mPrevChunk[QtileDim * (1 + SHARED_MEM_PADDING_CHUNK)];
  // init lPrevChunk (QTileDim x 1) in shared memory for previous rowsum of
  // cTile * dTile
  scalar_t *lPrevChunk =
      (scalar_t *)&lChunk[QtileDim * (1 + SHARED_MEM_PADDING_CHUNK)];
  // init nChunk (QTileDim x 1) in shared memory for normalizer state
  scalar_t *nChunk =
      (scalar_t *)&lPrevChunk[QtileDim * (1 + SHARED_MEM_PADDING_CHUNK)];
  // init nPrevChunk (QTileDim x 1) in shared memory for previous normalizer
  // state
  scalar_t *nPrevChunk =
      (scalar_t *)&nChunk[QtileDim * (1 + SHARED_MEM_PADDING_CHUNK)];
  // init fTileCol (QTileDim x 1) in shared memory for forget gate (first column
  // of the QtileDim x KVtileDim dTile)
  float *fTileCol =
      (float *)&nPrevChunk[QtileDim * (1 + SHARED_MEM_PADDING_CHUNK)];

  //! THREAD / WARP SETUP
  const uint warpId = threadIdx.x / WARP_SIZE;
  const uint laneId = threadIdx.x % WARP_SIZE;

  //? for loading Q, K, V from global memory
  const uint rowGmemTidxY = threadIdx.x / GMEM_LOAD_WARP_COLS_X;
  const uint colGmemTidxX = threadIdx.x % GMEM_LOAD_WARP_COLS_X;

  //? flatten the threads to 1D
  const uint flatThreadIdx = blockDim.x * threadIdx.y + threadIdx.x;

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
      // stream the qTile into shared memory
      // x-axis: dimHeads (laneId), y-axis: QtileDim (warpId)
      // we stream in sizeof(float4) = 16 bytes = 8 bfloat16 or float16 at a
      // time organize each warp in a 2D grid (8x4)
      // WARP_GROUP_SIZE = 4 warps load a 32x32 x 2byte tile
      // NUM_WARP_GROUPS = 2 groups load a 64x32 x 2byte tile
      // slide from left to right (inner), top to bottom (outer)
      const uint qWarpTileYEnd = CEIL_DIV(QtileDim, GMEM_LOAD_BLOCK_ROWS_Y);
      const uint qWarpTileXEnd = CEIL_DIV(dimHeads, GMEM_LOAD_BLOCK_2BITEMS_X);
      for (uint qWarpTileYIdx = 0; qWarpTileYIdx < qWarpTileYEnd;
           ++qWarpTileYIdx) {
        for (uint qWarpTileXIdx = 0; qWarpTileXIdx < qWarpTileXEnd;
             ++qWarpTileXIdx) {
          //? qWarpTileIdxes
          //* shared memory:
          const uint qWarpBlockSharedMemYIdx =
              GMEM_LOAD_BLOCK_ROWS_Y * qWarpTileYIdx + rowGmemTidxY;
          // Note: the WarpBlockSharedMemXIdx is left out as we add it
          // after casting to float4 to have the correct offset
          const uint qWarpBlockSharedMemXIdx =
              GMEM_LOAD_BLOCK_2BITEMS_X * qWarpTileXIdx;

          // pointer arithmetics: compute the pointer to the block in shared
          // and global mem in scalar_t (float16, bfloat16) dtype
          // TODO then cast to float4 for streaming to shared memory

          scalar_t *qTileWarpBlockSharedMemPtr =
              (scalar_t *)qTile +
              (dimHeads + SHARED_MEM_PADDING_TILE) * qWarpBlockSharedMemYIdx +
              qWarpBlockSharedMemXIdx;

          // no memory padding for global memory:
          scalar_t *matQTileWarpBlockGlobalMemPtr =
              (scalar_t *)matQ + qTileBlockGlobalMemIdx +
              (dimHeads)*qWarpBlockSharedMemYIdx + qWarpBlockSharedMemXIdx;

          *(((float4 *)(qTileWarpBlockSharedMemPtr)) + colGmemTidxX) =
              *(((float4 *)(matQTileWarpBlockGlobalMemPtr)) + colGmemTidxX);

#ifdef DEBUG_GMEMSTREAM1
          const uint qwGmemIdxY =
              (dimHeads * GMEM_LOAD_BLOCK_ROWS_Y) * qWarpTileYIdx +
              dimHeads * rowGmemTidxY;
          const uint qwGmemIdxYprint =
              (GMEM_LOAD_BLOCK_ROWS_Y)*qWarpTileYIdx + rowGmemTidxY;
          const uint qwGmemIdxX =
              GMEM_LOAD_BLOCK_2BITEMS_X * qWarpTileXIdx + colGmemTidxX;
          const uint qwGmemIdx =
              qTileBlockGlobalMemIdx + qwGmemIdxY + qwGmemIdxX;
          if ((blockIdx.x == 0) && (blockIdx.y == 0) &&
              ((threadIdx.x == 0) || (threadIdx.x == 1) || (threadIdx.x == 8) ||
               (threadIdx.x == 9))) {
            printf(
                "Bxy(%d,%d),Tidx:%d,w-l:%d-%d|rgTidXY(%d,%d)|qIdx(%d): "
                "qWTLoopXY(%d,%d),qWTSmemXY(%d,%d): gmemXY(%d,%d)=%f, "
                "smemXY(%d,%d)=%f, gmemptr=%f\n",
                blockIdx.x, blockIdx.y, threadIdx.x, warpId, laneId,
                colGmemTidxX, rowGmemTidxY, qTileIdx, qWarpTileXIdx,
                qWarpTileYIdx, qWarpBlockSharedMemXIdx, qWarpBlockSharedMemYIdx,
                qwGmemIdxX, qwGmemIdxYprint, type2float(matQ[qwGmemIdx]),
                qWarpBlockSharedMemXIdx + colGmemTidxX, qWarpBlockSharedMemYIdx,
                type2float(SMEMARRAY(qTile, dimHeads, qWarpBlockSharedMemYIdx,
                                     qWarpBlockSharedMemXIdx + colGmemTidxX)),
                type2float(*matQTileWarpBlockGlobalMemPtr));
          }
#endif
        }
      }
      __syncthreads(); // TODO: necessary?

#ifdef DEBUG_GMEMSTREAM2
      //! DEBUG: write hTile to global memory (has the same memory index as
      //! qTile)
      for (uint qWarpTileYIdx = 0; qWarpTileYIdx < qWarpTileYEnd;
           ++qWarpTileYIdx) {
        for (uint qWarpTileXIdx = 0; qWarpTileXIdx < qWarpTileXEnd;
             ++qWarpTileXIdx) {
          //? qWarpTileIdxes
          //* shared memory:
          const uint qWarpBlockSharedMemYIdx =
              GMEM_LOAD_BLOCK_ROWS_Y * qWarpTileYIdx + rowGmemTidxY;

          const uint qWarpBlockSharedMemXIdx =
              GMEM_LOAD_BLOCK_2BITEMS_X * qWarpTileXIdx;

          scalar_t *qTileWarpBlockSharedMemPtr =
              (scalar_t *)qTile +
              (dimHeads + SHARED_MEM_PADDING_TILE) * qWarpBlockSharedMemYIdx +
              qWarpBlockSharedMemXIdx;

          // no memory padding for global memory:
          scalar_t *matHTileWarpBlockGlobalMemPtr =
              (scalar_t *)matH + qTileBlockGlobalMemIdx +
              (dimHeads)*qWarpBlockSharedMemYIdx + qWarpBlockSharedMemXIdx;

          *(((float4 *)(matHTileWarpBlockGlobalMemPtr)) + colGmemTidxX) =
              *(((float4 *)(qTileWarpBlockSharedMemPtr)) + colGmemTidxX);
        }
      }
      __syncthreads(); // TODO: necessary?
#endif

      // TODO make this more efficient by combining PrevChunk and Chunk
      // and then iterate over 2*QTileDim at once
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

#ifdef DEBUG_gridSize1
        if ((blockIdx.x == 0) && (blockIdx.y == 1) && (threadIdx.x == 0) &&
            threadIdx.y == 0) {
          printf(
              "BIdx=(%d,%d): qTileIdx=%d, kvTileIdx=%d, cTileBlockXY=(%d,%d)\n",
              blockIdx.x, blockIdx.y, qTileIdx, kvTileIdx, cTileBlockXIdx,
              cTileBlockYIdx);
        }
#endif

#ifdef INCL_DMAT_COMP
        //! construct fTileCol for dTile computation (do only
        //! of kvTileIdx=0)
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
          //   const uint fChunkEnd = gridDim.y * qTileIdx + blockIdx.y + 1;
          //   const uint fChunkStart = gridDim.y * qTileIdx;
          //   max(0, gridDim.y * (qTileIdx - 1) + blockIdx.y + 1);
          const uint fChunkAccIterEnd = gridDim.y; // blockIdx.y + 1;
          for (uint fChunkAccIterIdx = 0; fChunkAccIterIdx < fChunkAccIterEnd;
               ++fChunkAccIterIdx) {

#ifdef DEBUG_fcolval2
            if ((blockIdx.x == 0) && (blockIdx.y == 0) && flatThreadIdx <= 1) {
              printf("IDX: cTileBlockY=%d, qTileIdx=%d, fChunkAccIterIdx=%d "
                     "(<%d), "
                     "blockIdx.y=%d, "
                     "FTIdx=%d, kvTileIdx=%d, InitFTileColVal=%f, "
                     "fTileColLast=%f\n",
                     cTileBlockYIdx, qTileIdx, fChunkAccIterIdx,
                     fChunkAccIterEnd, blockIdx.y, flatThreadIdx, kvTileIdx,
                     SMEMVECTOR(fTileCol, 3), SMEMVECTOR(fTileColLast, 0));
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
                // Debugging only: without logsigmoid
                // SMEMVECTOR(fChunk, fThreadSharedMemYIdx) =
                //     fGatePreact[fThreadGlobalMemIdx];
#ifdef DEBUG_fcolval5
                if ((blockIdx.x == 0) && (blockIdx.y == 0) &&
                    (flatThreadIdx <= 3)) {
                  printf("qTileIdx=%d, fChunkAccIterIdx=%d (<%d), "
                         "fWarpChunkIdx=%d,"
                         "blockIdx.y=%d, "
                         "flatThreadIdx=%d: "
                         "logsig(fgs)SRAM[%d]=%f, fgsHBM[%d]=%f, "
                         "logsig(fgs)HBM[%d]=%f\n",
                         qTileIdx, fChunkAccIterIdx, fChunkAccIterEnd,
                         fWarpChunkIdx, blockIdx.y, flatThreadIdx,
                         fThreadSharedMemYIdx,
                         type2float(SMEMVECTOR(fChunk, fThreadSharedMemYIdx)),
                         fThreadGlobalMemIdx,
                         type2float(fGatePreact[fThreadGlobalMemIdx]),
                         fThreadGlobalMemIdx,
                         type2float(
                             logsigmoid_g(fGatePreact[fThreadGlobalMemIdx])));
                }
#endif
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
#ifdef DEBUG_fcolval4
                  if ((blockIdx.x == 0) && (blockIdx.y == 0) &&
                      (flatThreadIdx == 7)) {
                    printf("qTileIdx=%d, fChunkAccIterIdx=%d (<%d), "
                           "blockIdx.y=%d, "
                           "flatThreadIdx=%d: "
                           "i=%d, dTileThreadYIdx=%d, f_acc=%f, fg(%d)=%f\n",
                           qTileIdx, fChunkAccIterIdx, fChunkEnd, blockIdx.y,
                           flatThreadIdx, i, dTileThreadYIdx, f_acc, fSumIdx,
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
            // Debugging only: without logsigmoid:
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
#endif // INCL_DMAT_COMP
       //! kTile Loading (into kvTile)
       // see description for qTile loading
       // loops over rows (outer) and columns (inner) of kTile
        const uint kvWarpTileYEnd = CEIL_DIV(KVtileDim, GMEM_LOAD_BLOCK_ROWS_Y);
        const uint kvWarpTileXEnd =
            CEIL_DIV(dimHeads, GMEM_LOAD_BLOCK_2BITEMS_X);
        for (uint kvWarpTileYIdx = 0; kvWarpTileYIdx < kvWarpTileYEnd;
             ++kvWarpTileYIdx) {
          for (uint kvWarpTileXIdx = 0; kvWarpTileXIdx < kvWarpTileXEnd;
               ++kvWarpTileXIdx) {
            //? kvWarpTileIdxes
            //* shared memory:
            const uint kvWarpBlockSharedMemYIdx =
                GMEM_LOAD_BLOCK_ROWS_Y * kvWarpTileYIdx + rowGmemTidxY;
            // Note: the WarpBlockSharedMemXIdx is left out as we add it
            // after casting to float4 to have the correct offset
            const uint kvWarpBlockSharedMemXIdx =
                GMEM_LOAD_BLOCK_2BITEMS_X * kvWarpTileXIdx;

            // pointer arithmetics: compute the pointer to the block in shared
            // and global mem in scalar_t (float16, bfloat16) dtype
            // TODO then cast to float4 for streaming to shared memory

            scalar_t *kvTileWarpBlockSharedMemPtr =
                (scalar_t *)kvTile +
                (dimHeads + SHARED_MEM_PADDING_TILE) *
                    kvWarpBlockSharedMemYIdx +
                kvWarpBlockSharedMemXIdx;

            // no memory padding for global memory:
            scalar_t *matKVTileWarpBlockGlobalMemPtr =
                (scalar_t *)matK + kvTileBlockGlobalMemIdx +
                (dimHeads)*kvWarpBlockSharedMemYIdx + kvWarpBlockSharedMemXIdx;

            *(((float4 *)(kvTileWarpBlockSharedMemPtr)) + colGmemTidxX) =
                *(((float4 *)(matKVTileWarpBlockGlobalMemPtr)) + colGmemTidxX);

#ifdef DEBUG_GMEMSTREAM3
            const uint kvwGmemIdxY =
                (dimHeads * GMEM_LOAD_BLOCK_ROWS_Y) * kvWarpTileYIdx +
                dimHeads * rowGmemTidxY;
            const uint kvwGmemIdxYprint =
                (GMEM_LOAD_BLOCK_ROWS_Y)*kvWarpTileYIdx + rowGmemTidxY;
            const uint kvwGmemIdxX =
                GMEM_LOAD_BLOCK_2BITEMS_X * kvWarpTileXIdx + colGmemTidxX;
            const uint kvwGmemIdx =
                kvTileBlockGlobalMemIdx + kvwGmemIdxY + kvwGmemIdxX;
            if ((blockIdx.x == 0) && (blockIdx.y == 0) &&
                ((threadIdx.x == 0) || (threadIdx.x == 1) ||
                 (threadIdx.x == 8) || (threadIdx.x == 9))) {
              printf("Bxy(%d,%d),Tidx:%d,w-l:%d-%d|rgTidXY(%d,%d)|qIdx(%d),"
                     "kvIdx(%d): "
                     "kvWTLoopXY(%d,%d),kvWTSmemXY(%d,%d): gmemXY(%d,%d)=%f, "
                     "smemXY(%d,%d)=%f, gmemptr=%f\n",
                     blockIdx.x, blockIdx.y, threadIdx.x, warpId, laneId,
                     colGmemTidxX, rowGmemTidxY, qTileIdx, kvTileIdx,
                     kvWarpTileXIdx, kvWarpTileYIdx, kvWarpBlockSharedMemXIdx,
                     kvWarpBlockSharedMemYIdx, kvwGmemIdxX, kvwGmemIdxYprint,
                     type2float(matK[kvwGmemIdx]),
                     kvWarpBlockSharedMemXIdx + colGmemTidxX,
                     kvWarpBlockSharedMemYIdx,
                     type2float(
                         SMEMARRAY(kvTile, dimHeads, kvWarpBlockSharedMemYIdx,
                                   kvWarpBlockSharedMemXIdx + colGmemTidxX)),
                     type2float(*matKVTileWarpBlockGlobalMemPtr));
            }
#endif
          }
        }
        __syncthreads(); // TODO: necessary?

#ifdef DEBUG_GMEMSTREAM4
        //! DEBUG: write kvTile to global memory matH
        if (kvTileIdx == 0 && blockIdx.y == 0) {
          for (uint kvWarpTileYIdx = 0; kvWarpTileYIdx < kvWarpTileYEnd;
               ++kvWarpTileYIdx) {
            for (uint kvWarpTileXIdx = 0; kvWarpTileXIdx < kvWarpTileXEnd;
                 ++kvWarpTileXIdx) {
              //? kvWarpTileIdxes
              //* shared memory:
              const uint kvWarpBlockSharedMemYIdx =
                  GMEM_LOAD_BLOCK_ROWS_Y * kvWarpTileYIdx + rowGmemTidxY;
              // Note: the WarpBlockSharedMemXIdx is left out as we add it
              // after casting to float4 to have the correct offset
              const uint kvWarpBlockSharedMemXIdx =
                  GMEM_LOAD_BLOCK_2BITEMS_X * kvWarpTileXIdx;

              // pointer arithmetics: compute the pointer to the block in shared
              // and global mem in scalar_t (float16, bfloat16) dtype
              // TODO then cast to float4 for streaming to shared memory

              scalar_t *kvTileWarpBlockSharedMemPtr =
                  (scalar_t *)kvTile +
                  (dimHeads + SHARED_MEM_PADDING_TILE) *
                      kvWarpBlockSharedMemYIdx +
                  kvWarpBlockSharedMemXIdx;

              // no memory padding for global memory:
              scalar_t *matHTileWarpBlockGlobalMemPtr =
                  (scalar_t *)matH + kvTileBlockGlobalMemIdx +
                  (dimHeads)*kvWarpBlockSharedMemYIdx +
                  kvWarpBlockSharedMemXIdx;

              *(((float4 *)(matHTileWarpBlockGlobalMemPtr)) + colGmemTidxX) =
                  *(((float4 *)(kvTileWarpBlockSharedMemPtr)) + colGmemTidxX);
            }
          }
          __syncthreads(); // TODO: necessary?
        }
#endif

#ifdef COMPUTE_QK_TENSORCORE
        //! compute S = (Q x K^T) with Tensor Cores
        // (QtileDim,KVtileDim) = (QtileDim,dimHeads) x (dimHeads,KVtileDim)

        // Four looplevels: QTileDim, dimHeads, KVtileDim
        // KVtileDim outer: loop over columns of kTile over a warpgroup
        // -> Note: we need this outermost loop as for 8 warps we cannot fit the
        // whole KV rows into registers
        // dimHeads: loop over columns of qTile and rows of kTile
        // QTileDim: loop over rows of qTile
        // KVtileDim inner: loop over columns of kTile within a warpgroup
        const uint kvDimOuterEnd =
            CEIL_DIV(KVtileDim, KVTC_DIM * WARP_GROUP_SIZE);
        const uint qDimEnd = CEIL_DIV(QtileDim, QTC_DIM * NUM_WARPS);
        const uint kvDimInnerEnd = WARP_GROUP_SIZE;
        const uint dimHeadsEnd = CEIL_DIV(dimHeads, KTC_DIM);

        // TODO from here
        // kvdim outer loop

        // S fragment accumulators
        nv::wmma::fragment<nv::wmma::accumulator, QTC_DIM, KVTC_DIM, KTC_DIM,
                           float>
            s[NUM_WARPS];

        // TODO where this
        // dimheads loops

        // qDim loop

        // kvDim inner loop

#else

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
                                SMEMARRAY(kvTile, dimHeads,
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

#endif // COMPUTE_QK_TENSORCORE
#ifdef OUTPUT_matS
        //! DEBUG only: write sTile to global memory
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
                SMEMARRAY(cTile, KVtileDim, cdWarpTileThreadSharedMemYIdx,
                          cdWarpTileThreadSharedMemXIdx);
          }
        }
#endif

#ifdef INCL_DMAT_COMP
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
            scalar_t m_val_unbounded = SMEMVECTOR(mChunk, lThreadSharedMemYIdx);
            // bound m_val from below to avoid overflow in exp(-m_val)
            // TODO: adapt -10 according to precision exp(10) = 22026.5, for
            // fp32 and bfloat16 this value could be higher
            scalar_t m_val_bounded =
                max_g(m_val_unbounded, float2type<scalar_t>(-10.0f));
            scalar_t m_val = max_g(m_prev_val, m_val_bounded);
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
#ifdef DEBUG_hsout2
            if ((blockIdx.x == 0) && (blockIdx.y == 0) &&
                (lThreadSharedMemYIdx < 4)) {
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
#endif // INCL_DMAT_COMP

        //! vTile Loading (into kvTile)
        // see description for qTile loading
        // loops over rows (outer) and columns (inner) of kTile
        for (uint kvWarpTileYIdx = 0; kvWarpTileYIdx < kvWarpTileYEnd;
             ++kvWarpTileYIdx) {
          for (uint kvWarpTileXIdx = 0; kvWarpTileXIdx < kvWarpTileXEnd;
               ++kvWarpTileXIdx) {
            //? kvWarpTileIdxes
            //* shared memory:
            const uint kvWarpBlockSharedMemYIdx =
                GMEM_LOAD_BLOCK_ROWS_Y * kvWarpTileYIdx + rowGmemTidxY;
            // Note: the WarpBlockSharedMemXIdx is left out as we add it
            // after casting to float4 to have the correct offset
            const uint kvWarpBlockSharedMemXIdx =
                GMEM_LOAD_BLOCK_2BITEMS_X * kvWarpTileXIdx;

            // pointer arithmetics: compute the pointer to the block in shared
            // and global mem in scalar_t (float16, bfloat16) dtype
            // TODO then cast to float4 for streaming to shared memory

            scalar_t *kvTileWarpBlockSharedMemPtr =
                (scalar_t *)kvTile +
                (dimHeads + SHARED_MEM_PADDING_TILE) *
                    kvWarpBlockSharedMemYIdx +
                kvWarpBlockSharedMemXIdx;

            // no memory padding for global memory:
            scalar_t *matKVTileWarpBlockGlobalMemPtr =
                (scalar_t *)matV + kvTileBlockGlobalMemIdx +
                (dimHeads)*kvWarpBlockSharedMemYIdx + kvWarpBlockSharedMemXIdx;

            *(((float4 *)(kvTileWarpBlockSharedMemPtr)) + colGmemTidxX) =
                *(((float4 *)(matKVTileWarpBlockGlobalMemPtr)) + colGmemTidxX);

#ifdef DEBUG_GMEMSTREAM5
            const uint kvwGmemIdxY =
                (dimHeads * GMEM_LOAD_BLOCK_ROWS_Y) * kvWarpTileYIdx +
                dimHeads * rowGmemTidxY;
            const uint kvwGmemIdxYprint =
                (GMEM_LOAD_BLOCK_ROWS_Y)*kvWarpTileYIdx + rowGmemTidxY;
            const uint kvwGmemIdxX =
                GMEM_LOAD_BLOCK_2BITEMS_X * kvWarpTileXIdx + colGmemTidxX;
            const uint kvwGmemIdx =
                kvTileBlockGlobalMemIdx + kvwGmemIdxY + kvwGmemIdxX;
            if ((blockIdx.x == 0) && (blockIdx.y == 0) &&
                ((threadIdx.x == 0) || (threadIdx.x == 1) ||
                 (threadIdx.x == 8) || (threadIdx.x == 9))) {
              printf("Bxy(%d,%d),Tidx:%d,w-l:%d-%d|rgTidXY(%d,%d)|qIdx(%d),"
                     "kvIdx(%d): "
                     "kvWTLoopXY(%d,%d),kvWTSmemXY(%d,%d): gmemXY(%d,%d)=%f, "
                     "smemXY(%d,%d)=%f, gmemptr=%f\n",
                     blockIdx.x, blockIdx.y, threadIdx.x, warpId, laneId,
                     colGmemTidxX, rowGmemTidxY, qTileIdx, kvTileIdx,
                     kvWarpTileXIdx, kvWarpTileYIdx, kvWarpBlockSharedMemXIdx,
                     kvWarpBlockSharedMemYIdx, kvwGmemIdxX, kvwGmemIdxYprint,
                     type2float(matK[kvwGmemIdx]),
                     kvWarpBlockSharedMemXIdx + colGmemTidxX,
                     kvWarpBlockSharedMemYIdx,
                     type2float(
                         SMEMARRAY(kvTile, dimHeads, kvWarpBlockSharedMemYIdx,
                                   kvWarpBlockSharedMemXIdx + colGmemTidxX)),
                     type2float(*matKVTileWarpBlockGlobalMemPtr));
            }
#endif
          }
        }
        __syncthreads(); // TODO: necessary?

#ifdef DEBUG_GMEMSTREAM6
        //! DEBUG: write kvTile to global memory matH
        if (kvTileIdx == 0 && blockIdx.y == 0) {
          for (uint kvWarpTileYIdx = 0; kvWarpTileYIdx < kvWarpTileYEnd;
               ++kvWarpTileYIdx) {
            for (uint kvWarpTileXIdx = 0; kvWarpTileXIdx < kvWarpTileXEnd;
                 ++kvWarpTileXIdx) {
              //? kvWarpTileIdxes
              //* shared memory:
              const uint kvWarpBlockSharedMemYIdx =
                  GMEM_LOAD_BLOCK_ROWS_Y * kvWarpTileYIdx + rowGmemTidxY;
              // Note: the WarpBlockSharedMemXIdx is left out as we add it
              // after casting to float4 to have the correct offset
              const uint kvWarpBlockSharedMemXIdx =
                  GMEM_LOAD_BLOCK_2BITEMS_X * kvWarpTileXIdx;

              // pointer arithmetics: compute the pointer to the block in shared
              // and global mem in scalar_t (float16, bfloat16) dtype
              // TODO then cast to float4 for streaming to shared memory

              scalar_t *kvTileWarpBlockSharedMemPtr =
                  (scalar_t *)kvTile +
                  (dimHeads + SHARED_MEM_PADDING_TILE) *
                      kvWarpBlockSharedMemYIdx +
                  kvWarpBlockSharedMemXIdx;

              // no memory padding for global memory:
              scalar_t *matHTileWarpBlockGlobalMemPtr =
                  (scalar_t *)matH + kvTileBlockGlobalMemIdx +
                  (dimHeads)*kvWarpBlockSharedMemYIdx +
                  kvWarpBlockSharedMemXIdx;

              *(((float4 *)(matHTileWarpBlockGlobalMemPtr)) + colGmemTidxX) =
                  *(((float4 *)(kvTileWarpBlockSharedMemPtr)) + colGmemTidxX);
            }
          }
          __syncthreads(); // TODO: necessary?
        }
#endif
        // TODO implement S += S V here (with tensor cores)

#ifdef INCL_DMAT_COMP
        // TODO bring back this later
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
              div_g(mul_g(exp_g(sub_g(m_prev_val, m_val)), n_prev_val), n_val);

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
                              c_val, SMEMARRAY(kvTile, dimHeads, i,
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
            // __syncthreads();

#ifdef DEBUG_hsout1
            if ((blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0) &&
                (threadIdx.y <= 8) && //(hWarpTileYIdx == 0) &&
                (hWarpTileXIdx == 0)) {
              printf(
                  "qTIdx=%d, kvTdx=%d, "
                  "cTBXY=(%d,%d), hWTXY=(%d,%d), hsXY[%d,%d]=%f, "
                  "sv_acc=%f, wf_hprev=%f, n_val=%f, n_prev_val=%f, m_val=%f, "
                  "m_prev_val=%f\n",
                  qTileIdx, kvTileIdx, cTileBlockXIdx, cTileBlockYIdx,
                  hWarpTileXIdx, hWarpTileYIdx, hWarpTileThreadSharedMemXIdx,
                  hWarpTileThreadSharedMemYIdx,
                  type2float(SMEMARRAY(hTile, dimHeads,
                                       hWarpTileThreadSharedMemYIdx,
                                       hWarpTileThreadSharedMemXIdx)),
                  sv_acc, type2float(weighting_factor_h_prev),
                  type2float(n_val), type2float(n_prev_val), type2float(m_val),
                  type2float(m_prev_val));
            }
#endif
          } // end hWarpTileXIdx
        }   // end hWarpTileYIdx
        __syncthreads();

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
        __syncthreads();
#endif // INCL_DMAT_COMP

      } // end looplevel 2: kvTileIdx

      // TODO sync all blocks here, necessary? The loop above has different
      // number of iterations for each block
      //   gridGroup.sync();

      //! Store hTile to global memory
      // see qTile write to global memory for reference
      for (uint qWarpTileYIdx = 0; qWarpTileYIdx < qWarpTileYEnd;
           ++qWarpTileYIdx) {
        for (uint qWarpTileXIdx = 0; qWarpTileXIdx < qWarpTileXEnd;
             ++qWarpTileXIdx) {
          //? qWarpTileIdxes
          //* shared memory:
          const uint qWarpBlockSharedMemYIdx =
              GMEM_LOAD_BLOCK_ROWS_Y * qWarpTileYIdx + rowGmemTidxY;

          const uint qWarpBlockSharedMemXIdx =
              GMEM_LOAD_BLOCK_2BITEMS_X * qWarpTileXIdx;

          scalar_t *hTileWarpBlockSharedMemPtr =
              (scalar_t *)hTile +
              (dimHeads + SHARED_MEM_PADDING_TILE) * qWarpBlockSharedMemYIdx +
              qWarpBlockSharedMemXIdx;

          // no memory padding for global memory:
          scalar_t *matHTileWarpBlockGlobalMemPtr =
              (scalar_t *)matH + qTileBlockGlobalMemIdx +
              (dimHeads)*qWarpBlockSharedMemYIdx + qWarpBlockSharedMemXIdx;

          *(((float4 *)(matHTileWarpBlockGlobalMemPtr)) + colGmemTidxX) =
              *(((float4 *)(hTileWarpBlockSharedMemPtr)) + colGmemTidxX);
        }
      }
      __syncthreads(); // TODO: necessary?

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
      __syncthreads();
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
  printf("NUM_WARPS:%d, GMEM_LOAD_BLOCK_COLS_X:%d, GMEM_LOAD_BLOCK_ROWS_Y:%d\n",
         NUM_WARPS, GMEM_LOAD_BLOCK_COLS_X, GMEM_LOAD_BLOCK_ROWS_Y);
  const int QtileDim = QTILE_DIM;   // blockdim for Q along seqLen dim
  const int KVtileDim = KVTILE_DIM; // blockdim for K&V along seqLen dim

  // kernel asserts
  if ((seqLen % QtileDim != 0) || (seqLen % KVtileDim != 0)) {
    printf("seqLen must be divisible by QblockDim and KVblockDim\n");
  }

  // determine the number of blocks and threads
  const dim3 blockDims(NUM_WARP_GROUPS * WARP_GROUP_SIZE * WARP_SIZE);

  // TODO: determine gridDims
  // Note @mbeck: should be dynamically allocated.
  // At first parallelize across batchSize and numHeads.
  // If more streaming multiprocessors available, parallelize across seqLen.
  //! NOTE: for now we only parallelize across batchSize and numHeads
  // TODO Need to dynamically check how many blocks we can launch
  // TODO add check if batchSize*numHeads exceeds max gridDim.x

  const dim3 gridDims(batchSize * numHeads, 1);
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
      sizeof(scalar_t) * QtileDim * (dimHeads + SHARED_MEM_PADDING_TILE);
  const uint kvTileSharedMemSize =
      sizeof(scalar_t) * KVtileDim * (dimHeads + SHARED_MEM_PADDING_TILE);
  const uint cdTileSharedMemSize =
      sizeof(scalar_t) * QtileDim * (KVtileDim + SHARED_MEM_PADDING_TILE);

  // See here:
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses
  // the idea of the padding is that every number is stored in a different
  // memory bank this should help to avoid bank conflicts as many threads need
  // to access the same input and forget gate values at the same time for the
  // gate matrix computation
  // TODO check if this is really helping!
  const uint iChunkSharedMemSize =
      sizeof(scalar_t) * KVtileDim * (1 + SHARED_MEM_PADDING_CHUNK);
  const uint fChunkSharedMemSize =
      sizeof(scalar_t) * QtileDim * (1 + SHARED_MEM_PADDING_CHUNK);

  // we keep these as float as it acts as accumulator
  const uint fTileColSharedMemSize =
      sizeof(float) * QtileDim * (1 + SHARED_MEM_PADDING_CHUNK);

  const uint nmlChunkSharedMemSize =
      sizeof(scalar_t) * QtileDim * (1 + SHARED_MEM_PADDING_CHUNK);

  // Input/Output tiles: 4x for qTile, vTile, kTile, hTile
  // Intermediate tiles: 2x for cTile, dTile
  // Intermediate tiles: 2x for mChunk, lChunk
  const uint sharedMemorySize =
      2 * qhTileSharedMemSize + 1 * kvTileSharedMemSize +
      2 * cdTileSharedMemSize + iChunkSharedMemSize + fChunkSharedMemSize +
      fTileColSharedMemSize + 6 * nmlChunkSharedMemSize;

  printf("blocksxy: %d-%d, threadsxy: %d-%d, QtileDim: %d, KVtileDim: %d, "
         "shared_mem in bytes: %d\n",
         gridDims.x, gridDims.y, blockDims.x, blockDims.y, QTILE_DIM, KVtileDim,
         sharedMemorySize);
  // cudaSetDevice(0);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  //* allocate intermediate HBM memory for D matrix (forget gate
  // preactivations)
  const uint fTileColLastGlobalMemSize = sizeof(float) * batchSize * numHeads;
  float *fTileColLast;
  gpuErrchk(cudaMalloc((void **)&fTileColLast, fTileColLastGlobalMemSize));
  gpuErrchk(cudaMemset(fTileColLast, 0, fTileColLastGlobalMemSize));

  auto kernel = kernels::vlstm_fw<scalar_t, QtileDim, KVtileDim>;
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

} // namespace vlstm