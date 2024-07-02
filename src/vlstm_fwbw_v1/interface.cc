// Copyright JKU Linz 2024
// Author: Maximilian Beck

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "../util/support.h"
#include "interface.h"
#include "kernel_dispatchers.h"
#include <cassert>

#define QTILE_DIM 8 // QtileDim: TileDim for Q along seqLen dim

namespace vlstm {

void interface::vlstm_fw(Tensor matH, Tensor vecN, Tensor vecM, Tensor matC,
                         Tensor matQ, Tensor matK, Tensor matV, Tensor vecIgp,
                         Tensor vecFgp) {
  const auto batchSize = matQ.size(0);
  const auto numHeads = matQ.size(1);
  const auto seqLen = matQ.size(2);
  const auto dimHeads = matQ.size(3);

  // checks: matQ & matK & matV should have the same shape
  // TODO: for later: enable different qk dim and v dim
  if (!(matK.size(0) == batchSize && matV.size(0) == batchSize)) {
    printf("matK & matV should have the same batch size!\n");
  }
  if (!(matK.size(1) == numHeads && matV.size(1) == numHeads)) {
    printf("matK & matV should have the same number of heads!\n");
  }
  if (!(matK.size(2) == seqLen && matV.size(2) == seqLen)) {
    printf("matK & matV should have the same sequence length!\n");
  }
  if (!(matK.size(3) == dimHeads && matV.size(3) == dimHeads)) {
    printf("matK & matV should have the same dimension of heads!\n");
  }
  if (!(vecIgp.size(0) == batchSize && vecIgp.size(1) == numHeads &&
        vecIgp.size(2) == seqLen)) {
    printf("vecIgp batch size, number of heads or "
           "sequence length mismatch!\n");
  }
  if (!(vecFgp.size(0) == batchSize && vecFgp.size(1) == numHeads &&
        vecFgp.size(2) == seqLen)) {
    printf("vecFgp batch size, number of heads or "
           "sequence length mismatch!\n");
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF2(
      matQ.scalar_type(), "vLSTMFw", ([&] {
        if (std::is_same<scalar_t, at::BFloat16>::value) {
          printf("before kernel dispatch - bfloat16!\n");
          kernel_dispatchers::vlstm_fw_dispatch<__nv_bfloat16>(
              reinterpret_cast<__nv_bfloat16 *>(matH.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(vecN.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(vecM.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(matC.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(matQ.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(matK.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(matV.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(vecIgp.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(vecFgp.data_ptr<scalar_t>()),
              batchSize, numHeads, seqLen, dimHeads);
        } else if (std::is_same<scalar_t, at::Half>::value) {
          printf("before kernel dispatch - float16!\n");
          kernel_dispatchers::vlstm_fw_dispatch<__half>(
              reinterpret_cast<__half *>(matH.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(vecN.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(vecM.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(matC.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(matQ.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(matK.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(matV.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(vecIgp.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(vecFgp.data_ptr<scalar_t>()),
              batchSize, numHeads, seqLen, dimHeads);
        } else if (std::is_same<scalar_t, float>::value) {
          printf("before kernel dispatch - float32!\n");
          kernel_dispatchers::vlstm_fw_dispatch<float>(
              reinterpret_cast<float *>(matH.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(vecN.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(vecM.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(matC.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(matQ.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(matK.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(matV.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(vecIgp.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(vecFgp.data_ptr<scalar_t>()), batchSize,
              numHeads, seqLen, dimHeads);
        } else {
          printf("No kernel for this dtype available.\n");
        }
      }));

  return;
}

void interface::vlstm_bw(Tensor matDeltaQ, Tensor matDeltaK, Tensor matDeltaV,
                         Tensor vecDeltaIgp, Tensor vecDeltaFgp, Tensor matC,
                         Tensor deltaH, Tensor matQ, Tensor matK, Tensor matV,
                         Tensor vecIgp, Tensor vecFgp, Tensor vecN,
                         Tensor vecM) {

  const auto batchSize = matQ.size(0);
  const auto numHeads = matQ.size(1);
  const auto seqLen = matQ.size(2);
  const auto dimHeads = matQ.size(3);

  // checks: matQ & matK & matV should have the same shape
  // TODO: for later: enable different qk dim and v dim
  if (!(matK.size(0) == batchSize && matV.size(0) == batchSize)) {
    printf("matK & matV should have the same batch size!\n");
  }
  if (!(matK.size(1) == numHeads && matV.size(1) == numHeads)) {
    printf("matK & matV should have the same number of heads!\n");
  }
  if (!(matK.size(2) == seqLen && matV.size(2) == seqLen)) {
    printf("matK & matV should have the same sequence length!\n");
  }
  if (!(matK.size(3) == dimHeads && matV.size(3) == dimHeads)) {
    printf("matK & matV should have the same dimension of heads!\n");
  }
  if (!(vecIgp.size(0) == batchSize && vecIgp.size(1) == numHeads &&
        vecIgp.size(2) == seqLen)) {
    printf("vecIgp batch size, number of heads or "
           "sequence length mismatch!\n");
  }
  if (!(vecFgp.size(0) == batchSize && vecFgp.size(1) == numHeads &&
        vecFgp.size(2) == seqLen)) {
    printf("vecFgp batch size, number of heads or "
           "sequence length mismatch!\n");
  }

  // // the output deltaErrors
  // auto matDeltaQ =
  //     torch::zeros({batchSize, numHeads, seqLen, dimHeads}, matQ.options());
  // auto matDeltaK =
  //     torch::zeros({batchSize, numHeads, seqLen, dimHeads}, matK.options());
  // auto matDeltaV =
  //     torch::zeros({batchSize, numHeads, seqLen, dimHeads}, matV.options());
  // auto vecDeltaIgp =
  //     torch::zeros({batchSize, numHeads, seqLen, 1}, vecIgp.options());
  // auto vecDeltaFgp =
  //     torch::zeros({batchSize, numHeads, seqLen, 1}, vecFgp.options());

  // //* unused for now (remove later), we allocate the memory directly in the
  // // kernel *//
  // // intermediate global memory allocations:
  // // intermediate cumsums for cumsum(deltaDtilde Tile)
  // // TODO check if there is some "fast part" of the global memory
  // // TODO make this allocation in the kernel call (without torch api, e.g.
  // // cuda_malloc())
  // // -> there we know the QTILE_DIM
  // // TODO error -> these should always be float32
  // auto vecDeltaDcumsum = torch::zeros({batchSize, numHeads, seqLen},
  //                                     matQ.options()); //
  // cumsum(deltaDtilde) const uint gridDimY = 2;
  // auto vecDeltaDcumsumChunkArr =
  //     torch::ones({batchSize, numHeads, gridDimY, QTILE_DIM},
  //                 matQ.options()); // cumsum(deltaDtilde)

  // // only for debugging: C or D matrix (S x S) (will be removed later)
  // auto matC =
  //     torch::zeros({batchSize, numHeads, seqLen, seqLen}, matQ.options());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF2(
      matQ.scalar_type(), "vLSTMBw", ([&] {
        if (std::is_same<scalar_t, at::BFloat16>::value) {
          printf("before kernel dispatch - bfloat16!\n");
          kernel_dispatchers::vlstm_bw_dispatch<__nv_bfloat16>(
              reinterpret_cast<__nv_bfloat16 *>(matDeltaQ.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(matDeltaK.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(matDeltaV.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(
                  vecDeltaIgp.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(
                  vecDeltaFgp.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(matC.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(deltaH.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(matQ.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(matK.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(matV.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(vecIgp.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(vecFgp.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(vecN.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(vecM.data_ptr<scalar_t>()),
              batchSize, numHeads, seqLen, dimHeads);
        } else if (std::is_same<scalar_t, at::Half>::value) {
          printf("before kernel dispatch - float16!\n");
          kernel_dispatchers::vlstm_bw_dispatch<__half>(
              reinterpret_cast<__half *>(matDeltaQ.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(matDeltaK.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(matDeltaV.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(vecDeltaIgp.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(vecDeltaFgp.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(matC.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(deltaH.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(matQ.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(matK.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(matV.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(vecIgp.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(vecFgp.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(vecN.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(vecM.data_ptr<scalar_t>()), batchSize,
              numHeads, seqLen, dimHeads);
        } else if (std::is_same<scalar_t, float>::value) {
          printf("before kernel dispatch - float32!\n");
          kernel_dispatchers::vlstm_bw_dispatch<float>(
              reinterpret_cast<float *>(matDeltaQ.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(matDeltaK.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(matDeltaV.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(vecDeltaIgp.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(vecDeltaFgp.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(matC.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(deltaH.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(matQ.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(matK.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(matV.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(vecIgp.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(vecFgp.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(vecN.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(vecM.data_ptr<scalar_t>()), batchSize,
              numHeads, seqLen, dimHeads);
        } else {
          printf("No kernel for this dtype available.\n");
        }
      }));

  return;
}

} // namespace vlstm

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vlstm_fw", &vlstm::interface::vlstm_fw, "vLSTM forward pass.");
  m.def("vlstm_bw", &vlstm::interface::vlstm_bw, "vLSTM backward pass.");
}