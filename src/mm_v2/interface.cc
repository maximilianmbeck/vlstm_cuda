
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "../util/support.h"
#include "interface.h"
#include "kernel_dispatchers.h"

namespace vlstm {

Tensor interface::mmkernel(Tensor matA, Tensor matB, Tensor matC) {
  const auto m = matA.size(0);
  const auto k = matA.size(1);
  const auto n = matB.size(1);
  auto matD = torch::zeros({m, n}, matC.options());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF2(
      matA.scalar_type(), "mmkernel", ([&] {
        if (std::is_same<scalar_t, at::BFloat16>::value) {
          printf("before kernel dispatch - bfloat16!\n");
          kernel_dispatchers::mmkernel_dispatch<__nv_bfloat16>(
              reinterpret_cast<float *>(matD.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(matA.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(matB.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(matC.data_ptr<scalar_t>()), m, n, k);
        } else if (std::is_same<scalar_t, at::Half>::value) {
          printf("before kernel dispatch - float16!\n");
          kernel_dispatchers::mmkernel_dispatch<__half>(
              reinterpret_cast<float *>(matD.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(matA.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(matB.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(matC.data_ptr<scalar_t>()), m, n, k);

        } else {
          printf("No kernel for this dtype available.\n");
        }
      }));

  return matD;
}

} // namespace vlstm

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("mmkernel", &vlstm::interface::mmkernel, "A mm kernel.");
}