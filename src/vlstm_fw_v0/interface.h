// Copyright JKU Linz 2023
// Author: Maximilian Beck

#pragma once

namespace vlstm {

using torch::Tensor;

namespace interface {

Tensor vlstm_fw(Tensor mat_Q, Tensor mat_K, Tensor mat_V);

} // namespace interface

} // namespace vlstm