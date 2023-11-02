// Copyright 2023 JKU Linz, All Rights Reserved
// Author: Korbinian Pöppel
// Adapted from the haste library
//
// See:
// Copyright 2020 LMNT, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#include "blas.h"
#include "inline_print.cuh"

cublasStatus_t cublasHgemv2(cublasHandle_t handle, cublasOperation_t trans,
                            int m, int n, const __half *alpha, const __half *A,
                            int lda, const __half *x, int incx,
                            const __half *beta, __half *y, int incy) {
  float alpha_f = __half2float(*alpha);
  float beta_f = __half2float(*beta);
  return cublasGemmEx(handle, trans, CUBLAS_OP_N, m, 1, n, &alpha_f, A,
                      CUDA_R_16F, m, x, CUDA_R_16F, n, &beta_f, y, CUDA_R_16F,
                      m, CUBLAS_COMPUTE_32F_FAST_16F,
                      CUBLAS_GEMM_DFALT_TENSOR_OP);
}

cublasStatus_t cublasHgemv3(cublasHandle_t handle, cublasOperation_t trans,
                            int m, int n, const __half *alpha, const __half *A,
                            int lda, const __half *x, int incx,
                            const __half *beta, __half *y, int incy) {
  return cublasGemmEx(handle, trans, CUBLAS_OP_N, m, 1, n, &alpha, A,
                      CUDA_R_16F, m, x, CUDA_R_16F, n, &beta, y, CUDA_R_16F, m,
                      CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DFALT_TENSOR_OP);
}

cublasStatus_t cublasHgemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n, const __half *alpha, const __half *A,
                           int lda, const __half *x, int incx,
                           const __half *beta, __half *y, int incy) {
  return cublasHgemm(handle, trans, CUBLAS_OP_N, m, 1, n, alpha, A, lda, x, n,
                     beta, y, m);
}

cublasStatus_t cublasHgemm2(cublasHandle_t handle, cublasOperation_t transa,
                            cublasOperation_t transb, int m, int n, int k,
                            const __half *alpha, /* host or device pointer */
                            const __half *A, int lda, const __half *B, int ldb,
                            const __half *beta, /* host or device pointer */
                            __half *C, int ldc) {
  float alpha_f = __half2float(*alpha);
  float beta_f = __half2float(*beta);
  return cublasGemmEx(handle, transa, transb, m, n, k, &alpha_f, A, CUDA_R_16F,
                      lda, B, CUDA_R_16F, ldb, &beta_f, C, CUDA_R_16F, ldc,
                      CUBLAS_COMPUTE_32F_FAST_16F, // Compute type
                      CUBLAS_GEMM_DFALT_TENSOR_OP  // Use Tensor Cores
  );
}

cublasStatus_t cublasHgemm3(cublasHandle_t handle, cublasOperation_t transa,
                            cublasOperation_t transb, int m, int n, int k,
                            const __half *alpha, /* host or device pointer */
                            const __half *A, int lda, const __half *B, int ldb,
                            const __half *beta, /* host or device pointer */
                            __half *C, int ldc) {
  return cublasGemmEx(handle, transa, transb, m, n, k, &alpha, A, CUDA_R_16F,
                      lda, B, CUDA_R_16F, ldb, &beta, C, CUDA_R_16F, ldc,
                      CUBLAS_COMPUTE_16F,         // Compute type
                      CUBLAS_GEMM_DFALT_TENSOR_OP // Use Tensor Cores
  );
}

template <typename T> __global__ void initKernel(T *data, int size, T value) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    data[idx] = value;
  }
}

void initVector_d(cudaStream_t stream, double *data, int size, double value) {
  int blockSize = 256;
  int numBlocks = (size + blockSize - 1) / blockSize;
  initKernel<<<numBlocks, blockSize, 0, stream>>>(data, size, value);
}

void initVector_f(cudaStream_t stream, float *data, int size, float value) {
  int blockSize = 256;
  int numBlocks = (size + blockSize - 1) / blockSize;
  initKernel<<<numBlocks, blockSize, 0, stream>>>(data, size, value);
}

void initVector_h(cudaStream_t stream, __half *data, int size, __half value) {
  int blockSize = 256;
  int numBlocks = (size + blockSize - 1) / blockSize;
  initKernel<<<numBlocks, blockSize, 0, stream>>>(data, size, value);
}

#if CUDART_VERSION >= 11020
#include <cuda_bf16.h>

void initVector_b(cudaStream_t stream, __nv_bfloat16 *data, int size,
                  __nv_bfloat16 value) {
  int blockSize = 256;
  int numBlocks = (size + blockSize - 1) / blockSize;
  initKernel<<<numBlocks, blockSize, 0, stream>>>(data, size, value);
}

cublasStatus_t cublasBgemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n, const __nv_bfloat16 *alpha,
                           const __nv_bfloat16 *A, int lda,
                           const __nv_bfloat16 *x, int incx,
                           const __nv_bfloat16 *beta, __nv_bfloat16 *y,
                           int incy) {
  float alpha_f = __bfloat162float(*alpha);
  float beta_f = __bfloat162float(*beta);
  return cublasGemmEx(handle, trans, CUBLAS_OP_N, m, 1, n, &alpha_f, A,
                      CUDA_R_16BF, m, x, CUDA_R_16BF, n, &beta_f, y,
                      CUDA_R_16BF, m, CUBLAS_COMPUTE_32F_FAST_16F,
                      CUBLAS_GEMM_DFALT_TENSOR_OP);
}

cublasStatus_t
cublasBgemm(cublasHandle_t handle, cublasOperation_t transa,
            cublasOperation_t transb, int m, int n, int k,
            const __nv_bfloat16 *alpha, /* host or device pointer */
            const __nv_bfloat16 *A, int lda, const __nv_bfloat16 *B, int ldb,
            const __nv_bfloat16 *beta, /* host or device pointer */
            __nv_bfloat16 *C, int ldc) {
  float alpha_f = __bfloat162float(*alpha);
  float beta_f = __bfloat162float(*beta);
  return cublasGemmEx(handle, transa, transb, m, n, k, &alpha_f, A, CUDA_R_16BF,
                      lda, B, CUDA_R_16BF, ldb, &beta_f, C, CUDA_R_16BF, ldc,
                      CUBLAS_COMPUTE_32F_FAST_16F, // Compute type
                      CUBLAS_GEMM_DFALT_TENSOR_OP  // Use Tensor Cores
  );
}

#endif
