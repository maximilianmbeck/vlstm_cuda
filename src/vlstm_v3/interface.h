#pragma once

// #include <torch/extension.h>

namespace vlstm {

using torch::Tensor;

namespace interface {

Tensor mmkernelv1(Tensor mat_A, Tensor mat_B);

Tensor qkvkernel(Tensor mat_Q, Tensor mat_K, Tensor mat_V);

} // namespace interface

} // namespace vlstm