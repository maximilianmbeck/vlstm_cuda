#pragma once

// #include <torch/extension.h>

namespace vlstm
{

    namespace kernels
    {

        void cudakernel2_dispatch(__nv_bfloat16 *mat_A, __nv_bfloat16 *mat_B, int rows, int cols);

    } // namespace kernels

} // namespace vlstm