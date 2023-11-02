#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "kernels.h"
#include "../util/support.h"

namespace vlstm
{

    namespace kernels
    {

        /*A kernel that copies from A to B*/
        __global__ void cudakernel2(__nv_bfloat16 *mat_A, __nv_bfloat16 *mat_B, int r, int c);

    } // namespace kernels


    __global__ void kernels::cudakernel2(__nv_bfloat16 *mat_A, __nv_bfloat16 *mat_B, int rdim, int cdim)
    {
        int cidx = blockIdx.x * blockDim.x + threadIdx.x;
        int ridx = blockIdx.y * blockDim.y + threadIdx.y;
        // printf("cidx: %d, ridx: %d\n", cidx, ridx);

        if (cidx < cdim && ridx < rdim)
        {
            int idx = ridx + cidx * rdim;
            mat_B[idx] = mat_A[idx];
        }
    }


    void kernels::cudakernel2_dispatch(__nv_bfloat16* mat_A, __nv_bfloat16 *mat_B, int rows, int cols) {
        printf("rows: %d, cols: %d\n", rows, cols);
        // determine the number of blocks and threads
        const dim3 block_threads(32, 32);
        const dim3 grid_blocks((rows + block_threads.y - 1) / block_threads.y, (cols + block_threads.x - 1) / block_threads.x);
        printf("blocksxy: %d-%d, threads: %d-%d\n", grid_blocks.x, grid_blocks.y, block_threads.x, block_threads.y);
        kernels::cudakernel2<<<grid_blocks, block_threads>>>(mat_A, mat_B, rows, cols);

    }

}