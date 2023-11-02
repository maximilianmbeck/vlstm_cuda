#pragma once

// #include <torch/extension.h>

namespace vlstm {

using torch::Tensor;

namespace interface {

Tensor testkernel(Tensor mat_A);

Tensor copykernel(Tensor mat_A);

Tensor mmkernelv1(Tensor mat_A, Tensor mat_B);

} // namespace interface

} // namespace vlstm