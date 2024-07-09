#pragma once

// #include <torch/extension.h>

namespace vlstm {

namespace kernel_dispatchers {

template <typename scalar_t>
void mmkernel_dispatch(float *matD, scalar_t *matA, scalar_t *matB, float *matC,
                       int m, int n, int k);

} // namespace kernel_dispatchers

} // namespace vlstm