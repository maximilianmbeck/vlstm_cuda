// Copyright 2023 IARAI GmbH, All Rights Reserved
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

#pragma once

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define AT __FILE__ ":" TOSTRING(__LINE__)

#if CUDART_VERSION >= 11020
#include <cuda_bf16.h>
#pragma message(AT " CUDART_VERSION with BF16: " TOSTRING(                     \
        CUDART_VERSION) ", CUDA_ARCH: " TOSTRING(__CUDA_ARCH__))
#else
#pragma message(AT " CUDART_VERSION: " TOSTRING(CUDART_VERSION))
#endif

// CONSTANTS
template <> __device__ __forceinline__ __nv_bfloat16 dscalar_three() {
  return __float2bfloat16(3.0f);
}

template <> __device__ __forceinline__ __nv_bfloat16 dscalar_two() {
  return __float2bfloat16(2.0f);
}

template <> __device__ __forceinline__ __nv_bfloat16 dscalar_one() {
  return __float2bfloat16(1.0f);
}

template <> __device__ __forceinline__ __nv_bfloat16 dscalar_half() {
  return __float2bfloat16(0.5f);
}

template <> __device__ __forceinline__ __nv_bfloat16 dscalar_zero() {
  return __float2bfloat16(0.0f);
}

template <> __forceinline__ __nv_bfloat16 scalar_one() {
  return __float2bfloat16(1.0f);
}

template <> __forceinline__ __nv_bfloat16 scalar_zero() {
  return __float2bfloat16(0.0f);
}

template <> __device__ __forceinline__ __nv_bfloat16 dscalar(double x) {
  return __float2bfloat16((float)x);
}

// -- CONSTANTS

// CONVERSIONS

template <> __device__ __forceinline__ float to_float(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

// -- CONVERSIONS

// ARITHMETIC FUNCTIONS
// ADD
template <>
__device__ __forceinline__ __nv_bfloat16 add_g(const __nv_bfloat16 a,
                                               const __nv_bfloat16 b) {
  return __hadd_rn(a, b);
}

// SUB
template <>
__device__ __forceinline__ __nv_bfloat16 sub_g(const __nv_bfloat16 a,
                                               const __nv_bfloat16 b) {
  return __hsub_rn(a, b);
}

// NEG
template <>
__device__ __forceinline__ __nv_bfloat16 neg_g(const __nv_bfloat16 a) {
  return __hneg(a);
}

// MUL
template <>
__device__ __forceinline__ __nv_bfloat16 mul_g(const __nv_bfloat16 a,
                                               const __nv_bfloat16 b) {
  return __hmul_rn(a, b);
}

// DIV
template <>
__device__ __forceinline__ __nv_bfloat16 div_g(const __nv_bfloat16 a,
                                               const __nv_bfloat16 b) {
  return __hdiv(a, b);
}

// -- ARITHMETIC FUNCTIONS

// COMPARISON OPERATIONS
template <>
__device__ __forceinline__ bool gt_g(const __nv_bfloat16 a,
                                     const __nv_bfloat16 b) {
  return __hgt(a, b);
}

template <>
__device__ __forceinline__ bool lt_g(const __nv_bfloat16 a,
                                     const __nv_bfloat16 b) {
  return __hgt(b, a);
}

template <> __device__ __forceinline__ bool gt_zero_g(const __nv_bfloat16 a) {
  return __hgt(a, __float2bfloat16(0.0f));
}

template <> __device__ __forceinline__ bool eq_zero_g(const __nv_bfloat16 a) {
  return __heq(a, __float2bfloat16(0.0f));
}

template <> __device__ __forceinline__ bool lt_zero_g(const __nv_bfloat16 a) {
  return __hgt(__float2bfloat16(0.0f), a);
}

// -- COMPARISON OPERATIONS

// Other functions
template <>
__device__ __forceinline__ __nv_bfloat16 exp_g(const __nv_bfloat16 x) {
  return hexp(x);
}

template <>
__device__ __forceinline__ __nv_bfloat16 log_g(const __nv_bfloat16 x) {
  return hlog(x);
}

template <>
__device__ __forceinline__ __nv_bfloat16 tanh_g(const __nv_bfloat16 x) {
  __nv_bfloat16 zero = dscalar_zero<__nv_bfloat16>();
  __nv_bfloat16 one = dscalar_one<__nv_bfloat16>();
  __nv_bfloat16 two = dscalar_two<__nv_bfloat16>();
  __nv_bfloat16 e2x;
  if (gt_g(x, zero)) {
    e2x = hexp(__hneg(__hmul(two, x)));
    return __hdiv(__hsub(one, e2x), __hadd(one, e2x));
  } else {
    e2x = hexp(__hmul(two, x));
    return __hdiv(__hsub(e2x, one), __hadd(one, e2x));
  }
}

template <>
__device__ __forceinline__ __nv_bfloat16 sigmoid_g(const __nv_bfloat16 x) {
  __nv_bfloat16 one = dscalar_one<__nv_bfloat16>();
  __nv_bfloat16 expx;
  if (gt_zero_g(x)) {
    return __hdiv(one, __hadd(one, hexp(__hneg(x))));
  } else {
    expx = hexp(x);
    return __hdiv(expx, __hadd(one, expx));
  }
}

template <>
__device__ __forceinline__ __nv_bfloat16
sigmoid_unstable_g(const __nv_bfloat16 x) {
  __nv_bfloat16 one = dscalar_one<__nv_bfloat16>();
  return __hdiv(one, __hadd(one, hexp(__hneg(x))));
}

template <>
__device__ __forceinline__ __nv_bfloat16 logsigmoid_g(const __nv_bfloat16 x) {
  __nv_bfloat16 one = dscalar_one<__nv_bfloat16>();
  if (gt_zero_g(x)) {
    return __hneg(hlog(__hadd(one, hexp(__hneg(x)))));
  } else {
    return __hsub(x, hlog(__hadd(one, hexp(x))));
  }
}

template <>
__device__ __forceinline__ __nv_bfloat16
d_sigmoid_g(const __nv_bfloat16 sigmoid_output) {
  return __hmul(sigmoid_output,
                __hsub(dscalar_one<__nv_bfloat16>(), sigmoid_output));
}

template <>
__device__ __forceinline__ __nv_bfloat16
d_tanh_g(const __nv_bfloat16 tanh_output) {
  return __hsub(dscalar_one<__nv_bfloat16>(), __hmul(tanh_output, tanh_output));
}

template <>
__device__ __forceinline__ __nv_bfloat16 max_g(const __nv_bfloat16 a,
                                               const __nv_bfloat16 b) {
  return __hmax(a, b);
}
