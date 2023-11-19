#pragma once

// #include <torch/extension.h>

namespace vlstm {

using torch::Tensor;

namespace interface {

Tensor qkvkernel(Tensor mat_Q, Tensor mat_K, Tensor mat_V);

} // namespace interface

} // namespace vlstm