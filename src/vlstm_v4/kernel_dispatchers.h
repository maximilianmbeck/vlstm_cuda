#pragma once

// #include <torch/extension.h>

namespace vlstm {

namespace kernel_dispatchers {

template <typename scalar_t>
void qkvkernel_dispatch(scalar_t *matC, scalar_t *matQ, scalar_t *matK,
                        scalar_t *matV, int batchSize, int numHeads, int seqLen,
                        int dimHeads);

} // namespace kernel_dispatchers

} // namespace vlstm