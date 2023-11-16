
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "../util/support.h"
#include "interface.h"
#include "kernel_dispatchers.h"
#include <cassert>

namespace vlstm {

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
              reinterpret_cast<__nv_bfloat16 *>(matB.data_ptr<scalar_t>()), m,
              n, k);
        } else if (std::is_same<scalar_t, at::Half>::value) {
          printf("before kernel dispatch - float16!\n");
          kernel_dispatchers::mmkernelv1_dispatch<__half>(
              reinterpret_cast<__half *>(matC.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(matA.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(matB.data_ptr<scalar_t>()), m, n, k);

        } else {
          printf("No kernel for this dtype available.\n");
        }
      }));

  return matC;
}

Tensor interface::qkvkernel(Tensor matQ, Tensor matK, Tensor matV) {
  const auto batchSize = matQ.size(0);
  const auto numHeads = matQ.size(1);
  const auto seqLen = matQ.size(2);
  const auto dimHeads = matQ.size(3);

  // checks: matK & matV should have the same shape
  if (!(matK.size(0) == batchSize && matV.size(0) == batchSize)) {
    printf("matK & matV should have the same batch size!\n");
  }
  if (!(matK.size(1) == numHeads && matV.size(1) == numHeads)) {
    printf("matK & matV should have the same number of heads!\n");
  }
  // Note matrix K is transposed
  if (!(matK.size(3) == seqLen && matV.size(2) == seqLen)) {
    printf("matK & matV should have the same sequence length! Did you forget "
           "to transpose K?\n");
  }
  if (!(matK.size(2) == dimHeads && matV.size(3) == dimHeads)) {
    printf("matK & matV should have the same dimension of heads! Did you "
           "forget to transpose K?\n");
  }

  auto matC =
      torch::zeros({batchSize, numHeads, seqLen, dimHeads}, matQ.options());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF2(
      matQ.scalar_type(), "qkvkernel", ([&] {
        if (std::is_same<scalar_t, at::BFloat16>::value) {
          printf("before kernel dispatch - bfloat16!\n");
          kernel_dispatchers::qkvkernel_dispatch<__nv_bfloat16>(
              reinterpret_cast<__nv_bfloat16 *>(matC.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(matQ.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(matK.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(matV.data_ptr<scalar_t>()),
              batchSize, numHeads, seqLen, dimHeads);
        } else if (std::is_same<scalar_t, at::Half>::value) {
          printf("before kernel dispatch - float16!\n");
          kernel_dispatchers::qkvkernel_dispatch<__half>(
              reinterpret_cast<__half *>(matC.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(matQ.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(matK.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(matV.data_ptr<scalar_t>()), batchSize,
              numHeads, seqLen, dimHeads);

        } else {
          printf("No kernel for this dtype available.\n");
        }
      }));

  return matC;
}

} // namespace vlstm

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mmkernelv1", &vlstm::interface::mmkernelv1, "A mm kernel.");
  m.def("qkvkernel", &vlstm::interface::qkvkernel, "A qkv kernel.");
}