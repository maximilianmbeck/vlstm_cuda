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

#if CUDART_VERSION >= 11000
#include <cuda_fp16.h>
#pragma message(AT " CUDART_VERSION with FP16: " TOSTRING(                     \
        CUDART_VERSION) ", CUDA_ARCH: " TOSTRING(__CUDA_ARCH__))
#else
#pragma message(AT " CUDART_VERSION: " TOSTRING(CUDART_VERSION))
#endif
#include <stdio.h>

// CONSTANTS
template <> __device__ __forceinline__ __half2 dscalar_three() {
  return __float2half2_rn(3.0f);
}

template <> __device__ __forceinline__ __half2 dscalar_two() {
  return __float2half2_rn(2.0f);
}

template <> __device__ __forceinline__ __half2 dscalar_one() {
  return __float2half2_rn(1.0f);
}

template <> __device__ __forceinline__ __half2 dscalar_half() {
  return __float2half2_rn(0.5f);
}

template <> __device__ __forceinline__ __half2 dscalar_zero() {
  return __float2half2_rn(0.0f);
}

template <> __forceinline__ __half2 scalar_one() {
  return __float2half2_rn(1.0f);
}

template <> __forceinline__ __half2 scalar_zero() {
  return __float2half2_rn(0.0f);
}

template <> __device__ __forceinline__ __half2 dscalar(double x) {
  return __float2half2_rn((float)x);
}

// -- CONSTANTS

// CONVERSIONS

// template <typename T> __device__ __forceinline__ float to_float(T x) {
//   return __half2float(x);
// }

// -- CONVERSIONS

// ARITHMETIC FUNCTIONS
// ADD
template <>
__device__ __forceinline__ __half2 add_g(const __half2 a, const __half2 b) {
  return __hadd2(a, b);
}

// SUB
template <>
__device__ __forceinline__ __half2 sub_g(const __half2 a, const __half2 b) {
  return __hsub2(a, b);
}

// NEG
template <> __device__ __forceinline__ __half2 neg_g(const __half2 a) {
  return __hneg2(a);
}

// MUL
template <>
__device__ __forceinline__ __half2 mul_g(const __half2 a, const __half2 b) {
  return __hmul2(a, b);
}

// DIV
template <>
__device__ __forceinline__ __half2 div_g(const __half2 a, const __half2 b) {
  return __h2div(a, b);
}

// ABS
template <> __device__ __forceinline__ __half2 abs_g(const __half2 a) {
  return __habs2(a);
}

// -- ARITHMETIC FUNCTIONS

// COMPARISON FUNCTIONS

template <> __device__ __forceinline__ bool eq_zero_g(const __half2 x) {
  return __hbeq2(x, dscalar_zero<__half2>());
}

// Other functions
template <> __device__ __forceinline__ __half2 exp_g(const __half2 x) {
  return h2exp(x);
}

template <> __device__ __forceinline__ __half2 log_g(const __half2 x) {
  return h2log(x);
}

template <> __device__ __forceinline__ __half2 tanh_g(const __half2 x) {
  const __half2 zero = dscalar_zero<__half2>();
  const __half2 one = dscalar_one<__half2>();
  const __half2 two = dscalar_two<__half2>();
  const __half szero = dscalar_zero<__half>();
  __half2 e2x;
  __half lowhalf = __low2half(x);
  __half highhalf = __high2half(x);
  if (__hbgt2(x, zero)) {
    // both x > 0
    e2x = h2exp(__hneg2(__hmul2(two, x)));
    e2x = __h2div(__hsub2(one, e2x), __hadd2(one, e2x));
  } else if (__hblt2(x, zero)) {
    //
    e2x = h2exp(__hmul2(two, x));
    e2x = __h2div(__hsub2(e2x, one), __hadd2(one, e2x));
  } else if (__hgt(lowhalf, szero)) {
    e2x = __halves2half2(__hneg(lowhalf), highhalf);
    e2x = h2exp(__hmul2(two, e2x));
    e2x = __h2div(__hsub2(e2x, one), __hadd2(one, e2x));
    lowhalf = __low2half(e2x);
    highhalf = __high2half(e2x);
    e2x = __halves2half2(__hneg(lowhalf), highhalf);
  } else {
    e2x = __halves2half2(lowhalf, __hneg(highhalf));
    e2x = h2exp(__hmul2(two, e2x));
    e2x = __h2div(__hsub2(e2x, one), __hadd2(one, e2x));
    lowhalf = __low2half(e2x);
    highhalf = __high2half(e2x);
    e2x = __halves2half2(lowhalf, __hneg(highhalf));
  }
  // Compare to __half only computation
  // lowhalf = __low2half(x);
  // highhalf = __high2half(x);
  // printf("Single runs: %f, %f\n", __half2float(tanh_g(lowhalf)),
  // __half2float(tanh_g(highhalf))); lowhalf = __low2half(e2x); highhalf =
  // __high2half(e2x); printf("Double run: %f, %f\n", __half2float(lowhalf),
  // __half2float(highhalf));
  return e2x;
}

template <>
__device__ __forceinline__ __half2 sigmoid_unstable_g(const __half2 x) {
  const __half2 one = dscalar_one<__half2>();
  return __h2div(one, __hadd2(one, h2exp(__hneg2(x))));
}

template <> __device__ __forceinline__ __half2 sigmoid_g(const __half2 x) {
  const __half2 zero = dscalar_zero<__half2>();
  const __half2 one = dscalar_one<__half2>();
  const __half szero = dscalar_zero<__half>();
  const __half sone = dscalar_one<__half>();
  __half2 expx;
  __half lowhalf = __low2half(x);
  __half highhalf = __high2half(x);
  // printf("Sigmoid: %f, %f\n", __half2float(lowhalf), __half2float(highhalf));

  if (__hbgt2(x, zero)) {
    // both x > 0
    // printf("Case 1: %f, %f\n", __half2float(lowhalf),
    // __half2float(highhalf));
    expx = h2exp(__hneg2(x));
    expx = __h2div(one, __hadd2(one, expx));
    lowhalf = __low2half(expx);
    highhalf = __high2half(expx);
    // printf("Case 1res: %f, %f\n", __half2float(lowhalf),
    // __half2float(highhalf));
    return expx;
  } else if (__hblt2(x, zero)) {
    //
    // printf("Case 2: %f, %f\n", __half2float(lowhalf),
    // __half2float(highhalf));
    expx = h2exp(x);
    expx = __h2div(expx, __hadd2(one, expx));
    lowhalf = __low2half(expx);
    highhalf = __high2half(expx);
    // printf("Case 2res: %f, %f\n", __half2float(lowhalf),
    // __half2float(highhalf));
    return expx;
  } else if (__hgt(lowhalf, szero)) {
    // printf("Case 3: %f, %f\n", __half2float(lowhalf),
    // __half2float(highhalf));
    expx = __halves2half2(__hneg(lowhalf), highhalf);
    expx = h2exp(expx);
    expx = __h2div(one, __hadd2(one, expx));
    lowhalf = __low2half(expx);
    highhalf = __high2half(expx);
    // printf("Case 3res: %f, %f\n", __half2float(lowhalf),
    // __half2float(__hsub(sone, highhalf)));
    return __halves2half2(lowhalf, __hsub(sone, highhalf));
  } else {
    // printf("Case 4: %f, %f\n", __half2float(lowhalf),
    // __half2float(highhalf));
    expx = __halves2half2(lowhalf, __hneg(highhalf));
    expx = h2exp(expx);
    expx = __h2div(one, __hadd2(one, expx));
    lowhalf = __low2half(expx);
    highhalf = __high2half(expx);
    // printf("Case 4res: %f, %f\n", __half2float(__hsub(sone, lowhalf)),
    // __half2float(highhalf));
    return __halves2half2(__hsub(sone, lowhalf), highhalf);
  }
}

template <> __device__ __forceinline__ __half2 logsigmoid_g(const __half2 x) {
  const __half2 zero = dscalar_zero<__half2>();
  const __half2 one = dscalar_one<__half2>();
  const __half szero = dscalar_zero<__half>();
  const __half sone = dscalar_one<__half>();
  __half2 expx;
  __half lowhalf = __low2half(x);
  __half highhalf = __high2half(x);
  // printf("Sigmoid: %f, %f\n", __half2float(lowhalf), __half2float(highhalf));

  if (__hbgt2(x, zero)) {
    expx = h2exp(__hneg2(x));
    expx = __hneg2(h2log(__hadd2(one, expx)));
    return expx;
  } else if (__hblt2(x, zero)) {
    expx = h2exp(x);
    expx = __hsub2(x, h2log(__hadd2(one, expx)));
    return expx;
  } else if (__hgt(lowhalf, szero)) {
    expx = __halves2half2(__hneg(lowhalf), highhalf);
    expx = h2exp(expx);
    expx = __hneg2(h2log(__hadd2(one, expx)));
    lowhalf = __low2half(expx);
    highhalf = __hadd(highhalf, __high2half(expx));
    return __halves2half2(lowhalf, highhalf);
  } else {
    expx = __halves2half2(lowhalf, __hneg(highhalf));
    expx = h2exp(expx);
    expx = __hneg2(h2log(__hadd2(one, expx)));
    lowhalf = __hadd(lowhalf, __low2half(expx));
    highhalf = __high2half(expx);
    return __halves2half2(lowhalf, highhalf);
  }
}

template <>
__device__ __forceinline__ __half2 d_sigmoid_g(const __half2 sigmoid_output) {
  return __hmul2(sigmoid_output,
                 __hsub2(dscalar_one<__half2>(), sigmoid_output));
}

template <>
__device__ __forceinline__ __half2 d_tanh_g(const __half2 tanh_output) {
  return __hsub2(dscalar_one<__half2>(), __hmul2(tanh_output, tanh_output));
}

template <>
__device__ __forceinline__ __half2 max_g(const __half2 a, const __half2 b) {
  return __hmax2(a, b);
}

template <>
__device__ __forceinline__ __half
low_half_2h<__half2, __half>(const __half2 x) {
  return __low2half(x);
}

template <>
__device__ __forceinline__ __half
high_half_2h<__half2, __half>(const __half2 x) {
  return __high2half(x);
}

template <> __device__ __forceinline__ bool gt_zero_g(const __half2 x) {
  return __hbgt2(x, __float2half2_rn(0.0f));
}

template <> __device__ __forceinline__ bool lt_zero_g(const __half2 x) {
  return __hblt2(x, __float2half2_rn(0.0f));
}

template <>
__device__ __forceinline__ bool low_half_gt_zero_2h(const __half2 x) {
  return __hgt(__low2half(x), __float2half(0.0f));
}

template <>
__device__ __forceinline__ bool high_half_gt_zero_2h(const __half2 x) {
  return __hgt(__low2half(x), __float2half(0.0f));
}

template <>
__device__ __forceinline__ __half2 join_halves_2h(const __half a,
                                                  const __half b) {
  return __halves2half2(a, b);
}
