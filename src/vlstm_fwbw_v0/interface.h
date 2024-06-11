// Copyright JKU Linz 2023
// Author: Maximilian Beck

#pragma once
#include <torch/torch.h>
#include <tuple>

namespace vlstm {

using torch::Tensor;

namespace interface {

std::tuple<Tensor, Tensor> vlstm_fw(Tensor mat_Q, Tensor mat_K, Tensor mat_V,
                                    Tensor iGatePreact, Tensor fGatePreact);

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
vlstm_bw(Tensor delta_H, Tensor mat_Q, Tensor mat_K, Tensor mat_V,
         Tensor iGatePreact, Tensor fGatePreact, Tensor vec_n, Tensor vec_m);

} // namespace interface

} // namespace vlstm