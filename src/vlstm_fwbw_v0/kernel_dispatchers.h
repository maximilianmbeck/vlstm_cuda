// Copyright JKU Linz 2024
// Author: Maximilian Beck

#pragma once

namespace vlstm {

namespace kernel_dispatchers {

template <typename scalar_t>
void vlstm_fw_dispatch(scalar_t *matH, scalar_t *vecN, scalar_t *vecM,
                       scalar_t *matC, scalar_t *matQ, scalar_t *matK,
                       scalar_t *matV, scalar_t *iGatePreact,
                       scalar_t *fGatePreact, int batchSize, int numHeads,
                       int seqLen, int dimHeads);

template <typename scalar_t>
void vlstm_bw_dispatch(scalar_t *deltaQ, scalar_t *deltaK, scalar_t *deltaV,
                       scalar_t *deltaIGatePreact, scalar_t *deltaFGatePreact,
                       scalar_t *deltaH, scalar_t *matQ, scalar_t *matK,
                       scalar_t *matV, scalar_t *iGatePreact,
                       scalar_t *fGatePreact, scalar_t *vecN, scalar_t *vecM,
                       int batchSize, int numHeads, int seqLen, int dimHeads);

} // namespace kernel_dispatchers

} // namespace vlstm