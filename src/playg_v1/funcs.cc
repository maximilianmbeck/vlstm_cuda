
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "interface.h"
#include "kernels.h"
#include "../util/support.h"

namespace vlstm
{

    Tensor interface::testkernel(Tensor mat_A)
    {
        printf("Test kernel!\n");
        return mat_A; 
    }

    Tensor interface::testkernel2(Tensor mat_A)
    {
        const auto rows = mat_A.size(0);
        const auto cols = mat_A.size(1);
        auto mat_B = torch::zeros_like(mat_A);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF2(mat_A.scalar_type(), "testkernel2", ([&]
                                                                                  {
                                                                                    //   bool isbfloat = std::is_same<scalar_t, at::BFloat16>::value;
                                                                                    //   printf("dtype is bfloat: %s", isbfloat);
                                                                                    printf("before kernel dispatch!\n"); 
                                                                                    kernels::cudakernel2_dispatch(
                                                                                        reinterpret_cast<__nv_bfloat16 *>(mat_A.data_ptr<scalar_t>()),
                                                                                        reinterpret_cast<__nv_bfloat16 *>(mat_B.data_ptr<scalar_t>()),
                                                                                        rows, cols); 
                                                                                    
                                                                                    }));
                                                                                  
        return mat_B;
    }


} // namespace vlstm

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("testkernel", &vlstm::interface::testkernel, "A test kernel.");
    m.def("testkernel2", &vlstm::interface::testkernel2, "A test kernel2.");
}