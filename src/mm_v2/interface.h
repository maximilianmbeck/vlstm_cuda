#pragma once

// #include <torch/extension.h>

namespace vlstm {

using torch::Tensor;

namespace interface {

Tensor mmkernel(Tensor matA, Tensor matB, Tensor matC);

} // namespace interface

} // namespace vlstm