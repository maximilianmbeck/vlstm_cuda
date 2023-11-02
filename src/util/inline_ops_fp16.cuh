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

#pragma once

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define AT __FILE__ ":" TOSTRING(__LINE__)

#if CUDART_VERSION >= 11000
#include <cuda_fp16.h>
#pragma message(AT " CUDART_VERSION with FP16: " TOSTRING(                     \
    CUDART_VERSION) ", CUDA_ARCH: " TOSTRING(__CUDA_ARCH__))
#else
#pragma message(AT " CUDART_VERSION: " TOSTRING(CUDART_VERSION))
#endif

// CONSTANTS
template <> __device__ __forceinline__ __half dscalar_three() {
  return __float2half(3.0f);
}

template <> __device__ __forceinline__ __half dscalar_two() {
  return __float2half(2.0f);
}

template <> __device__ __forceinline__ __half dscalar_one() {
  return __float2half(1.0f);
}

template <> __device__ __forceinline__ __half dscalar_half() {
  return __float2half(0.5f);
}

template <> __device__ __forceinline__ __half dscalar_zero() {
  return __float2half(0.0f);
}

template <> __forceinline__ __half scalar_one() { return __float2half(1.0f); }

template <> __forceinline__ __half scalar_zero() { return __float2half(0.0f); }

template <> __device__ __forceinline__ __half dscalar(double x) {
  return __float2half((float)x);
}

// -- CONSTANTS

// CONVERSIONS

template <> __device__ __forceinline__ float to_float(__half x) {
  return __half2float(x);
}

// -- CONVERSIONS

// ARITHMETIC FUNCTIONS
// ADD
template <>
__device__ __forceinline__ __half add_g(const __half a, const __half b) {
  return __hadd_rn(a, b);
}

// SUB
template <>
__device__ __forceinline__ __half sub_g(const __half a, const __half b) {
  return __hsub_rn(a, b);
}

// NEG
template <> __device__ __forceinline__ __half neg_g(const __half a) {
  return __hneg(a);
}

// MUL
template <>
__device__ __forceinline__ __half mul_g(const __half a, const __half b) {
  return __hmul_rn(a, b);
}

// DIV
template <>
__device__ __forceinline__ __half div_g(const __half a, const __half b) {
  return __hdiv(a, b);
}

// -- ARITHMETIC FUNCTIONS

// COMPARISON OPERATIONS
template <>
__device__ __forceinline__ bool gt_g(const __half a, const __half b) {
  return __hgt(a, b);
}

template <>
__device__ __forceinline__ bool lt_g(const __half a, const __half b) {
  return __hgt(b, a);
}

template <> __device__ __forceinline__ bool gt_zero_g(const __half a) {
  return __hgt(a, __float2half(0.0f));
}

template <> __device__ __forceinline__ bool lt_zero_g(const __half a) {
  return __hgt(__float2half(0.0f), a);
}

// -- COMPARISON OPERATIONS

// Other functions
template <> __device__ __forceinline__ __half exp_g(const __half x) {
  return hexp(x);
}

template <> __device__ __forceinline__ __half tanh_g(const __half x) {
  __half zero = dscalar_zero<__half>();
  __half one = dscalar_one<__half>();
  __half two = dscalar_two<__half>();
  __half e2x;
  if (gt_g(x, zero)) {
    e2x = hexp(__hneg(__hmul(two, x)));
    return __hdiv(__hsub(one, e2x), __hadd(one, e2x));
  } else {
    e2x = hexp(__hmul(two, x));
    return __hdiv(__hsub(e2x, one), __hadd(one, e2x));
  }
}

template <> __device__ __forceinline__ __half sigmoid_g(const __half x) {
  __half one = dscalar_one<__half>();
  __half expx;
  if (gt_zero_g(x)) {
    return __hdiv(one, __hadd(one, hexp(__hneg(x))));
  } else {
    expx = hexp(x);
    return __hdiv(expx, __hadd(one, expx));
  }
}

template <>
__device__ __forceinline__ __half sigmoid_unstable_g(const __half x) {
  __half one = dscalar_one<__half>();
  return __hdiv(one, __hadd(one, hexp(__hneg(x))));
}

template <>
__device__ __forceinline__ __half d_sigmoid_g(const __half sigmoid_output) {
  return __hmul(sigmoid_output, __hsub(dscalar_one<__half>(), sigmoid_output));
}

template <>
__device__ __forceinline__ __half d_tanh_g(const __half tanh_output) {
  return __hsub(dscalar_one<__half>(), __hmul(tanh_output, tanh_output));
}

template <>
__device__ __forceinline__ __half max_g(const __half a, const __half b) {
  return __hmax(a, b);
}
