// Copyright JKU Linz 2023
// Author: Maximilian Beck

#pragma once
#include <torch/torch.h>
#include <tuple>

namespace vlstm {

using torch::Tensor;

namespace interface {

void vlstm_fw(Tensor matH, Tensor vecN, Tensor vecM, Tensor matC, Tensor matQ,
              Tensor matK, Tensor matV, Tensor vecIgp, Tensor vecFgp);

void vlstm_bw(Tensor matDeltaQ, Tensor matDeltaK, Tensor matDeltaV,
              Tensor vecDeltaIg, Tensor vecDeltaFg, Tensor matC,
              Tensor matDeltaH, Tensor matQ, Tensor matK, Tensor matV,
              Tensor vecIgp, Tensor vecFgp, Tensor vecN, Tensor vecM);

} // namespace interface

} // namespace vlstm