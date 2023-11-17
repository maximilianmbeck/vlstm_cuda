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
#define DEBUG2 1
// #define DEBUG3 1

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

  cg::grid_group gridGroup = cg::this_grid();

  // Assign threads to x-y-coordinates for accessing shared memory tiles
  const uint tileX = blockIdx.x * blockDim.x + threadIdx.x;
  const uint tileY = blockIdx.y * blockDim.y + threadIdx.y;

  // Most outer loop: Loop over batchSize * numHeads (can be parallelized later
  // along gridDim.z)
  const uint batchHeadStep = seqLen * dimHeads;
  const uint batchHeadEnd = batchSize * numHeads * batchHeadStep;
  //! Note: batchHeadMemIdx indices into global memory
  for (uint batchHeadMemIdx = 0; batchHeadMemIdx < batchHeadEnd;
       batchHeadMemIdx += batchHeadStep) {

    // Ends for looplevel 1&2:
    uint qTileEnd = CEIL_DIV(seqLen, QtileDim);
    uint kvTileEnd = CEIL_DIV(seqLen, KVtileDim);
#ifdef DEBUG
    if ((blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0) &&
        (threadIdx.y == 0)) {
      printf("In Kernel: QtileDim: %d, KVtileDim: %d, TblockDim:%d\n", QtileDim,
             KVtileDim, TblockDim);
      printf("In Kernel: qTileEnd: %d, kvTileEnd: %d\n", qTileEnd, kvTileEnd);
    }
#endif
    // looplevel 1: loop over Qtile blocks along seqLen dim
    // Note: qTileIdx does not index into global memory
    for (uint qTileIdx = 0; qTileIdx < qTileEnd; ++qTileIdx) {

      // offset in Q matrix for qTile (global memory)
      // hint: blockDim.y * gridDim.y = QtileDim
      const uint qTileMemIdx =
          batchHeadMemIdx + qTileIdx * dimHeads * blockDim.y * gridDim.y;

      // init qTile in shared memory
      // TODO use dynamic shared memory!
      __shared__ scalar_t qTile[QtileDim][HD_SIZE];

      //! qTile Loading
      //? qcTileIdxes
      // left upper corner of qTileBlock in Q
      const uint qcTileBlockMemIdx = qTileMemIdx +
                                     dimHeads * TblockDim * blockIdx.y +
                                     TblockDim * blockIdx.x;
      const uint qcTileThreadMemIdx =
          qcTileBlockMemIdx + dimHeads * threadIdx.y + threadIdx.x;

      // We have enough threads to load the whole qTile into shared memory
      // load qTile into shared memory
      qTile[tileY][tileX] = matQ[qcTileThreadMemIdx];
      gridGroup.sync();

      // initialize result cTile in shared memory
      // __shared__ scalar_t cTile[QtileDim][HD_SIZE];
      scalar_t c_acc = dscalar_zero<scalar_t>();

      // looplevel 2: loop over KVtile blocks along seqLen dim
      for (uint kvTileIdx = 0; kvTileIdx < kvTileEnd; ++kvTileIdx) {

        // offset in K&V matrix for kTile & vTile (global memory)
        const uint kvTileMemIdx =
            batchHeadMemIdx + kvTileIdx * dimHeads * blockDim.y * gridDim.y;

        // init kTile and vTile in shared memory
        __shared__ scalar_t kTile[KVtileDim][HD_SIZE];
        __shared__ scalar_t vTile[KVtileDim][HD_SIZE];

        // init sTile (QtileDim x KVTileDim) in shared memory for intermediate
        // result of QK^T
        __shared__ scalar_t sTile[QtileDim][KVtileDim];

        //! kTile & vTile Loading
        //? kvTileIdxes
        // left upper corner of kTileBlock in K
        const uint kcTileBlockMemIdx = kvTileMemIdx +
                                       dimHeads * TblockDim * blockIdx.y +
                                       TblockDim * blockIdx.x;

        const uint kcTileThreadMemIdx =
            kcTileBlockMemIdx + dimHeads * threadIdx.y + threadIdx.x;

        // We have enough threads to load the whole kvTile into shared memory
        // load kTile into shared memory
        kTile[tileY][tileX] = matK[kcTileThreadMemIdx];
        // load vTile into shared memory
        vTile[tileY][tileX] = matV[kcTileThreadMemIdx];
        gridGroup.sync();
#ifdef DEBUG3
        printf("B<%d,%d>T<%d,%d> - kTile[%d][%d]: %f\n", blockIdx.x, blockIdx.y,
               threadIdx.x, threadIdx.y, tileY, tileX,
               type2float(kTile[tileY][tileX]));
#endif

        //! compute S = Q x K^T, i.e. fill sTile
        // (QtileDim,KVtileDim) = (QtileDim,dimHeads) x (dimHeads,KVtileDim)
        // each thread computes one entry in the sTile
        // TODO: What to do with the left over threads? (currently idle)
        // only use threads that fall into the sTile
        if ((tileY) < QtileDim && (tileX) < KVtileDim) {
#ifdef DEBUG3
          printf("B<%d,%d>T<%d,%d> - kTile[%d][%d]: %f\n", blockIdx.x,
                 blockIdx.y, threadIdx.x, threadIdx.y, tileY, tileX,
                 type2float(kTile[tileY][tileX]));
#endif
          scalar_t qk_acc = dscalar_zero<scalar_t>();
          for (uint i = 0; i < dimHeads; ++i) {
#ifdef DEBUG2
            if (tileX == 0 && tileY == 3) {
              printf("1-B<%d,%d>T<%d,%d> - qTile[%d][%d]: %f\n", blockIdx.x,
                     blockIdx.y, threadIdx.x, threadIdx.y, tileY, i,
                     type2float(qTile[tileY][i]));
              printf("1-B<%d,%d>T<%d,%d> - kTile[%d][%d]: %f\n", blockIdx.x,
                     blockIdx.y, threadIdx.x, threadIdx.y, tileX, i,
                     type2float(kTile[tileX][i]));
            }
            if (tileX == 0 && tileY == 4) {
              printf("2-B<%d,%d>T<%d,%d> - qTile[%d][%d]: %f\n", blockIdx.x,
                     blockIdx.y, threadIdx.x, threadIdx.y, tileY, i,
                     type2float(qTile[tileY][i]));
              printf("2-B<%d,%d>T<%d,%d> - kTile[%d][%d]: %f\n", blockIdx.x,
                     blockIdx.y, threadIdx.x, threadIdx.y, tileX, i,
                     type2float(kTile[tileX][i]));
            }
#endif

            qk_acc = add_g(qk_acc, mul_g(qTile[tileY][i], kTile[tileX][i]));
            // #ifdef DEBUG2
            //             if (tileX == 0 && tileY == 4) {
            //               printf("B<%d,%d>T<%d,%d> - kvTidx(%d) i(%d) qk_acc:
            //               %f\n",
            //                      blockIdx.x, blockIdx.y, threadIdx.x,
            //                      threadIdx.y, kvTileIdx, i,
            //                      type2float(qk_acc));
            //             }
            // #endif
          }
          sTile[tileY][tileX] = qk_acc;
#ifdef DEBUG3
          if (tileX == 0) {
            printf("B<%d,%d>T<%d,%d> - kvTidx(%d) sTile[%d][%d]: %f\n",
                   blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, kvTileIdx,
                   tileY, tileX, type2float(sTile[tileY][tileX]));
            printf("B<%d,%d>T<%d,%d> - kvTidx(%d) qTile[%d][%d]: %f\n",
                   blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, kvTileIdx,
                   tileY, tileX, type2float(qTile[tileY][tileX]));
            printf("B<%d,%d>T<%d,%d> - kvTidx(%d) kTile[%d][%d]: %f\n",
                   blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, kvTileIdx,
                   tileY, tileX, type2float(kTile[tileY][tileX]));
          }
#endif
        }
        gridGroup.sync();

        //! compute C += S * V, i.e. fill cTile
        // (QtileDim,dimHeads) = (QtileDim,KVtileDim) x (KVtileDim,dimHeads)
        // only use threads that fall into the cTile (should be all threads)
        if ((tileY) < QtileDim && (tileX) < dimHeads) {
          scalar_t sv_acc = dscalar_zero<scalar_t>();
          for (uint i = 0; i < KVtileDim; ++i) {
            sv_acc = add_g(sv_acc, mul_g(sTile[tileY][i], vTile[i][tileX]));
          }
          c_acc = add_g(c_acc, sv_acc);
          // cTile[blockIdx.y + threadIdx.y][blockIdx.x + threadIdx.x] =
          //     add_g(cTile[blockIdx.y + threadIdx.y][blockIdx.x +
          //     threadIdx.x],
          //           sv_acc);
        }
        gridGroup.sync();
        matC[qcTileThreadMemIdx] = kTile[tileY][tileX];
      }

      //! write cTile to global memory (has the same memory index as qTile)
      // matC[qcTileThreadMemIdx] = c_acc;
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

  const dim3 gridDims(CEIL_DIV(dimHeads, blockDims.x),
                      CEIL_DIV(QtileDim, blockDims.y));
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
  //     <<<gridDims, blockDims>>>(matC, matQ, matK, matV, batchSize, numHeads,
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