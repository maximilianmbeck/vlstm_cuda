#pragma once

// #include <torch/extension.h>

namespace vlstm
{

    namespace kernels
    {
        template <typename scalar_t>
        void copykernel_dispatch(scalar_t *mat_A, scalar_t *mat_B, int rows, int cols);
        
        // template <> void copykernel_dispatch<__nv_bfloat16>(__nv_bfloat16 *mat_A, __nv_bfloat16 *mat_B, int rows, int cols);

        // template <> void copykernel_dispatch<__half>(__half *mat_A, __half *mat_B, int rows, int cols);


        // template <typename scalar_t>
        // void mmkernelv1_dispatch(scalar_t *matC, scalar_t *matA,
        //                           scalar_t *matB, int m, int n, int k);


    } // namespace kernels

} // namespace vlstm