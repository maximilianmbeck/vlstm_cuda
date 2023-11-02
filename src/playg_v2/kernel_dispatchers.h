#pragma once

// #include <torch/extension.h>

namespace vlstm
{

    namespace kernels
    {

        void copykernel_dispatch(__nv_bfloat16 *mat_A, __nv_bfloat16 *mat_B, int rows, int cols);

        void mmkernelv1_dispatch(__nv_bfloat16 *matC, __nv_bfloat16 *matA,
                                  __nv_bfloat16 *matB, int m, int n, int k);

    } // namespace kernels

} // namespace vlstm