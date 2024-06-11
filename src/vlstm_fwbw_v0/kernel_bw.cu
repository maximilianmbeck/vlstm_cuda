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
        "In KKernel: gdim.x: %d, gdim.y: %d, gdim.z: %d, bdim.x: %d, bdim.y: "
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

  // init fTileColLast (1 x 1) in shared memory for forget gate (last row value
  // of fTileCol)
  float *fTileColLast =
      (float *)&lPrevChunk[QtileDim * (1 + SHARED_MEM_PADDING)];

  //! PARALLELIZE ALONG BATCHSIZE * NUMHEADS (gridDim.x)
  const uint batchHeadStepQKV = seqLen * dimHeads;
  const uint batchHeadStepIFgate = seqLen * 1;
  const uint batchHeadStepCD = seqLen * seqLen;
  const uint numBatchHeads = batchSize * numHeads;
  // End for looplevel 0:
  const uint batchHeadEnd = CEIL_DIV(numBatchHeads, gridDim.x);
  // looplevel 0: loop over batches and heads
  for (uint batchHeadIdx = 0; batchHeadIdx < batchHeadEnd; ++batchHeadIdx) {

    uint batchHeadGridXGlobalMemIdxQKV =
        (batchHeadStepQKV * gridDim.x) * batchHeadIdx +
        (batchHeadStepQKV)*blockIdx.x;

    uint batchHeadGridXGlobalMemIdxIFgate =
        (batchHeadStepIFgate * gridDim.x) * batchHeadIdx +
        (batchHeadStepIFgate)*blockIdx.x;

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
    SMEMVECTOR(fTileColLast, 0) =
        float2type<float>(0.0f); // could also just write 0.0f

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

    } // end looplevel 1
  }   // end looplevel 0
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
  const uint fTileColLastSharedMemSize =
      sizeof(float) * 1 * (1 + SHARED_MEM_PADDING);

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
      4 * qkvhTileSharedMemSize + 2 * cdTileSharedMemSize +
      iChunkSharedMemSize + fChunkSharedMemSize + fTileColSharedMemSize +
      2 * mChunkSharedMemSize + 2 * lChunkSharedMemSize +
      2 * nChunkSharedMemSize + fTileColLastSharedMemSize;

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