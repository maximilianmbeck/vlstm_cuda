
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "../util/support.h"
#include "interface.h"
#include "kernel_dispatchers.h"

namespace vlstm {

Tensor interface::testkernel(const Tensor& mat_A) {
  printf("Test kernel!\n");
  return mat_A;
}

Tensor interface::copykernel(const Tensor& mat_A) {
  const auto rows = mat_A.size(0);
  const auto cols = mat_A.size(1);
  auto mat_B = torch::zeros_like(mat_A);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF2(
      mat_A.scalar_type(), "copykernel", ([&] {
        if (std::is_same<scalar_t, at::BFloat16>::value) {
          printf("before kernel dispatch - bfloat16!\n");
          kernel_dispatchers::copykernel_dispatch<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16 *>(mat_A.data_ptr<scalar_t>()),
            reinterpret_cast<__nv_bfloat16 *>(mat_B.data_ptr<scalar_t>()), rows,
            cols);
        } else if (std::is_same<scalar_t, at::Half>::value) {
          printf("before kernel dispatch - float16!\n");
          kernel_dispatchers::copykernel_dispatch<__half>(
            reinterpret_cast<__half *>(mat_A.data_ptr<scalar_t>()),
            reinterpret_cast<__half *>(mat_B.data_ptr<scalar_t>()), rows,
            cols);

        } else {
          printf("No kernel for this dtype available.\n");
        }
      }));

  return mat_B;
}

Tensor interface::mmkernelv1(Tensor matA, Tensor matB) {
  const auto m = matA.size(0);
  const auto k = matA.size(1);
  const auto n = matB.size(1);
  auto matC = torch::zeros({m, n}, matA.options());
  
  AT_DISPATCH_FLOATING_TYPES_AND_HALF2(
          matA.scalar_type(), "mmkernelv1", ([&] {
            if (std::is_same<scalar_t, at::BFloat16>::value) {
              printf("before kernel dispatch - bfloat16!\n");
              kernel_dispatchers::mmkernelv1_dispatch<__nv_bfloat16>(
                reinterpret_cast<__nv_bfloat16 *>(matC.data_ptr<scalar_t>()), 
                reinterpret_cast<__nv_bfloat16 *>(matA.data_ptr<scalar_t>()),
                reinterpret_cast<__nv_bfloat16 *>(matB.data_ptr<scalar_t>()),
                m, n, k);
            } else if (std::is_same<scalar_t, at::Half>::value) {
              printf("before kernel dispatch - float16!\n");
              kernel_dispatchers::mmkernelv1_dispatch<__half>(
                reinterpret_cast<__half *>(matC.data_ptr<scalar_t>()), 
                reinterpret_cast<__half *>(matA.data_ptr<scalar_t>()),
                reinterpret_cast<__half *>(matB.data_ptr<scalar_t>()),
                m, n, k);

            } else {
              printf("No kernel for this dtype available.\n");
            }
          }));

  return matC;
}

} // namespace vlstm

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("testkernel", &vlstm::interface::testkernel, "A test kernel.");
  m.def("copykernel", &vlstm::interface::copykernel, "A copy kernel.");
  m.def("mmkernelv1", &vlstm::interface::mmkernelv1, "A mm kernel.");
}