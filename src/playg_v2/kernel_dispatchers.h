#pragma once

// #include <torch/extension.h>

namespace vlstm
{

    namespace kernels
    {

        void copykernel_dispatch(__nv_bfloat16 *mat_A, __nv_bfloat16 *mat_B, int rows, int cols);

        void mmkernelv1_dispatch(__nv_bfloat16 *mat_A, __nv_bfloat16 *mat_B, int wA, int wB);

    } // namespace kernels

} // namespace vlstm