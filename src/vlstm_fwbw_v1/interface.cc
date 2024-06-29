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

std::tuple<Tensor, Tensor, Tensor, Tensor>
interface::vlstm_fw(Tensor matQ, Tensor matK, Tensor matV, Tensor iGatePreact,
                    Tensor fGatePreact) {
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
  if (!(iGatePreact.size(0) == batchSize && iGatePreact.size(1) == numHeads &&
        iGatePreact.size(2) == seqLen)) {
    printf("iGatePreact batch size, number of heads or "
           "sequence length mismatch!\n");
  }
  if (!(fGatePreact.size(0) == batchSize && fGatePreact.size(1) == numHeads &&
        fGatePreact.size(2) == seqLen)) {
    printf("fGatePreact batch size, number of heads or "
           "sequence length mismatch!\n");
  }

  // the output matrix
  auto matH =
      torch::zeros({batchSize, numHeads, seqLen, dimHeads}, matQ.options());

  // store intermediate computations for backward pass
  // TODO these should be deallocated autmatically by pytorch
  auto vecN = torch::zeros({batchSize, numHeads, seqLen, 1}, matQ.options());
  auto vecM = torch::zeros({batchSize, numHeads, seqLen, 1}, matQ.options());

  // only for debugging: C or D matrix (S x S) (will be removed later)
  auto matC =
      torch::zeros({batchSize, numHeads, seqLen, seqLen}, matQ.options());

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
              reinterpret_cast<__nv_bfloat16 *>(
                  iGatePreact.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(
                  fGatePreact.data_ptr<scalar_t>()),
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
              reinterpret_cast<__half *>(iGatePreact.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(fGatePreact.data_ptr<scalar_t>()),
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
              reinterpret_cast<float *>(iGatePreact.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(fGatePreact.data_ptr<scalar_t>()),
              batchSize, numHeads, seqLen, dimHeads);
        } else {
          printf("No kernel for this dtype available.\n");
        }
      }));

  return std::make_tuple(matH, vecN, vecM, matC);
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>
interface::vlstm_bw(Tensor deltaH, Tensor matQ, Tensor matK, Tensor matV,
                    Tensor iGatePreact, Tensor fGatePreact, Tensor vecN,
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
  if (!(iGatePreact.size(0) == batchSize && iGatePreact.size(1) == numHeads &&
        iGatePreact.size(2) == seqLen)) {
    printf("iGatePreact batch size, number of heads or "
           "sequence length mismatch!\n");
  }
  if (!(fGatePreact.size(0) == batchSize && fGatePreact.size(1) == numHeads &&
        fGatePreact.size(2) == seqLen)) {
    printf("fGatePreact batch size, number of heads or "
           "sequence length mismatch!\n");
  }

  // the output deltaErrors
  auto deltaQ =
      torch::zeros({batchSize, numHeads, seqLen, dimHeads}, matQ.options());
  auto deltaK =
      torch::zeros({batchSize, numHeads, seqLen, dimHeads}, matK.options());
  auto deltaV =
      torch::zeros({batchSize, numHeads, seqLen, dimHeads}, matV.options());
  auto deltaIGatePreact =
      torch::zeros({batchSize, numHeads, seqLen, 1}, iGatePreact.options());
  auto deltaFGatePreact =
      torch::zeros({batchSize, numHeads, seqLen, 1}, fGatePreact.options());

  //* unused for now (remove later), we allocate the memory directly in the
  // kernel *//
  // intermediate global memory allocations:
  // intermediate cumsums for cumsum(deltaDtilde Tile)
  // TODO check if there is some "fast part" of the global memory
  // TODO make this allocation in the kernel call (without torch api, e.g.
  // cuda_malloc())
  // -> there we know the QTILE_DIM
  // TODO error -> these should always be float32
  auto csDeltaDTildeVec = torch::zeros({batchSize, numHeads, seqLen},
                                       matQ.options()); // cumsum(deltaDtilde)
  const uint gridDimY = 2;
  auto csDeltaDTildeChunkArr =
      torch::ones({batchSize, numHeads, gridDimY, QTILE_DIM},
                  matQ.options()); // cumsum(deltaDtilde)

  // only for debugging: C or D matrix (S x S) (will be removed later)
  auto matC =
      torch::zeros({batchSize, numHeads, seqLen, seqLen}, matQ.options());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF2(
      matQ.scalar_type(), "vLSTMBw", ([&] {
        if (std::is_same<scalar_t, at::BFloat16>::value) {
          printf("before kernel dispatch - bfloat16!\n");
          kernel_dispatchers::vlstm_bw_dispatch<__nv_bfloat16>(
              reinterpret_cast<__nv_bfloat16 *>(deltaQ.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(deltaK.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(deltaV.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(
                  deltaIGatePreact.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(
                  deltaFGatePreact.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(matC.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(deltaH.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(matQ.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(matK.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(matV.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(
                  iGatePreact.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(
                  fGatePreact.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(vecN.data_ptr<scalar_t>()),
              reinterpret_cast<__nv_bfloat16 *>(vecM.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(
                  csDeltaDTildeChunkArr.data_ptr<float>()),
              reinterpret_cast<float *>(csDeltaDTildeVec.data_ptr<float>()),
              batchSize, numHeads, seqLen, dimHeads);
        } else if (std::is_same<scalar_t, at::Half>::value) {
          printf("before kernel dispatch - float16!\n");
          kernel_dispatchers::vlstm_bw_dispatch<__half>(
              reinterpret_cast<__half *>(deltaQ.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(deltaK.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(deltaV.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(deltaIGatePreact.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(deltaFGatePreact.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(matC.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(deltaH.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(matQ.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(matK.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(matV.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(iGatePreact.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(fGatePreact.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(vecN.data_ptr<scalar_t>()),
              reinterpret_cast<__half *>(vecM.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(
                  csDeltaDTildeChunkArr.data_ptr<float>()),
              reinterpret_cast<float *>(csDeltaDTildeVec.data_ptr<float>()),
              batchSize, numHeads, seqLen, dimHeads);
        } else if (std::is_same<scalar_t, float>::value) {
          printf("before kernel dispatch - float32!\n");
          kernel_dispatchers::vlstm_bw_dispatch<float>(
              reinterpret_cast<float *>(deltaQ.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(deltaK.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(deltaV.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(deltaIGatePreact.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(deltaFGatePreact.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(matC.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(deltaH.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(matQ.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(matK.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(matV.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(iGatePreact.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(fGatePreact.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(vecN.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(vecM.data_ptr<scalar_t>()),
              reinterpret_cast<float *>(
                  csDeltaDTildeChunkArr.data_ptr<float>()),
              reinterpret_cast<float *>(csDeltaDTildeVec.data_ptr<float>()),
              batchSize, numHeads, seqLen, dimHeads);
        } else {
          printf("No kernel for this dtype available.\n");
        }
      }));

  return std::make_tuple(deltaQ, deltaK, deltaV, deltaIGatePreact,
                         deltaFGatePreact, matC, csDeltaDTildeChunkArr,
                         csDeltaDTildeVec);
}

} // namespace vlstm

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vlstm_fw", &vlstm::interface::vlstm_fw, "vLSTM forward pass.");
  m.def("vlstm_bw", &vlstm::interface::vlstm_bw, "vLSTM backward pass.");
}