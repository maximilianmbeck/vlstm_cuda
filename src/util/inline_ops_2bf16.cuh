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
#pragma message(AT " CUDART_VERSION with FP16: " TOSTRING(                     \
        CUDART_VERSION) ", CUDA_ARCH: " TOSTRING(__CUDA_ARCH__))
#else
#pragma message(AT " CUDART_VERSION: " TOSTRING(CUDART_VERSION))
#endif

// CONSTANTS
template <> __device__ __forceinline__ __nv_bfloat162 dscalar_three() {
  return __float2bfloat162_rn(3.0f);
}

template <> __device__ __forceinline__ __nv_bfloat162 dscalar_two() {
  return __float2bfloat162_rn(2.0f);
}

template <> __device__ __forceinline__ __nv_bfloat162 dscalar_one() {
  return __float2bfloat162_rn(1.0f);
}

template <> __device__ __forceinline__ __nv_bfloat162 dscalar_half() {
  return __float2bfloat162_rn(0.5f);
}

template <> __device__ __forceinline__ __nv_bfloat162 dscalar_zero() {
  return __float2bfloat162_rn(0.0f);
}

template <> __forceinline__ __nv_bfloat162 scalar_one() {
  return __float2bfloat162_rn(1.0f);
}

template <> __forceinline__ __nv_bfloat162 scalar_zero() {
  return __float2bfloat162_rn(0.0f);
}

template <> __device__ __forceinline__ __nv_bfloat162 dscalar(double x) {
  return __float2bfloat162_rn((float)x);
}

// -- CONSTANTS

// ARITHMETIC FUNCTIONS
// ADD
template <>
__device__ __forceinline__ __nv_bfloat162 add_g(const __nv_bfloat162 a,
                                                const __nv_bfloat162 b) {
  return __hadd2(a, b);
}

// SUB
template <>
__device__ __forceinline__ __nv_bfloat162 sub_g(const __nv_bfloat162 a,
                                                const __nv_bfloat162 b) {
  return __hsub2(a, b);
}

// NEG
template <>
__device__ __forceinline__ __nv_bfloat162 neg_g(const __nv_bfloat162 a) {
  return __hneg2(a);
}

// MUL
template <>
__device__ __forceinline__ __nv_bfloat162 mul_g(const __nv_bfloat162 a,
                                                const __nv_bfloat162 b) {
  return __hmul2(a, b);
}

// DIV
template <>
__device__ __forceinline__ __nv_bfloat162 div_g(const __nv_bfloat162 a,
                                                const __nv_bfloat162 b) {
  return __halves2bfloat162(__hdiv(__low2bfloat16(a), __low2bfloat16(b)),
                            __hdiv(__high2bfloat16(a), __high2bfloat16(b)));
}

// -- ARITHMETIC FUNCTIONS

// COMPARISON FUNCTIONS

template <> __device__ __forceinline__ bool eq_zero_g(const __nv_bfloat162 x) {
  return __hbeq2(x, dscalar_zero<__nv_bfloat162>());
}

// Other functions
template <>
__device__ __forceinline__ __nv_bfloat162 exp_g(const __nv_bfloat162 x) {
  return h2exp(x);
}

template <>
__device__ __forceinline__ __nv_bfloat162 log_g(const __nv_bfloat162 x) {
  return h2log(x);
}

template <>
__device__ __forceinline__ __nv_bfloat162 tanh_g(const __nv_bfloat162 x) {
  const __nv_bfloat162 zero = dscalar_zero<__nv_bfloat162>();
  const __nv_bfloat162 one = dscalar_one<__nv_bfloat162>();
  const __nv_bfloat162 two = dscalar_two<__nv_bfloat162>();
  const __nv_bfloat16 szero = dscalar_zero<__nv_bfloat16>();
  __nv_bfloat162 e2x;
  __nv_bfloat16 lowbfloat16 = __low2bfloat16(x);
  __nv_bfloat16 highbfloat16 = __high2bfloat16(x);
  if (__hbgt2(x, zero)) {
    // both x > 0
    e2x = h2exp(__hneg2(__hmul2(two, x)));
    e2x = div_g(__hsub2(one, e2x), __hadd2(one, e2x));
  } else if (__hblt2(x, zero)) {
    //
    e2x = h2exp(__hmul2(two, x));
    e2x = div_g(__hsub2(e2x, one), __hadd2(one, e2x));
  } else if (__hgt(lowbfloat16, szero)) {
    e2x = __halves2bfloat162(__hneg(lowbfloat16), highbfloat16);
    e2x = h2exp(__hmul2(two, e2x));
    e2x = div_g(__hsub2(e2x, one), __hadd2(one, e2x));
    lowbfloat16 = __low2bfloat16(e2x);
    highbfloat16 = __high2bfloat16(e2x);
    e2x = __halves2bfloat162(__hneg(lowbfloat16), highbfloat16);
  } else {
    e2x = __halves2bfloat162(lowbfloat16, __hneg(highbfloat16));
    e2x = h2exp(__hmul2(two, e2x));
    e2x = div_g(__hsub2(e2x, one), __hadd2(one, e2x));
    lowbfloat16 = __low2bfloat16(e2x);
    highbfloat16 = __high2bfloat16(e2x);
    e2x = __halves2bfloat162(lowbfloat16, __hneg(highbfloat16));
  }
  // lowbfloat16 = __low2bfloat16(x);
  // highbfloat16 = __high2bfloat16(x);
  // printf("Single runs: %f, %f\n", __bfloat162float(tanh_g(lowbfloat16)),
  // __bfloat162float(tanh_g(highbfloat16))); lowbfloat16 = __low2bfloat16(e2x);
  // highbfloat16 = __high2bfloat16(e2x);
  // printf("Double run: %f, %f\n", __bfloat162float(lowbfloat16),
  // __bfloat162float(highbfloat16));
  return e2x;
}

template <>
__device__ __forceinline__ __nv_bfloat162 sigmoid_g(const __nv_bfloat162 x) {
  const __nv_bfloat162 zero = dscalar_zero<__nv_bfloat162>();
  const __nv_bfloat162 one = dscalar_one<__nv_bfloat162>();
  const __nv_bfloat16 szero = dscalar_zero<__nv_bfloat16>();
  const __nv_bfloat16 sone = dscalar_one<__nv_bfloat16>();
  __nv_bfloat162 expx;
  __nv_bfloat16 lowhalf = __low2bfloat16(x);
  __nv_bfloat16 highhalf = __high2bfloat16(x);
  if (__hbgt2(x, zero)) {
    // both x > 0
    expx = h2exp(__hneg2(x));
    return __h2div(one, __hadd2(one, expx));
  } else if (__hblt2(x, zero)) {
    //
    expx = h2exp(x);
    return __h2div(expx, __hadd2(one, expx));
  } else if (__hgt(lowhalf, szero)) {
    expx = __halves2bfloat162(__hneg(lowhalf), highhalf);
    expx = h2exp(expx);
    expx = __h2div(one, __hadd2(one, expx));
    lowhalf = __low2bfloat16(expx);
    highhalf = __high2bfloat16(expx);
    return __halves2bfloat162(lowhalf, __hsub(sone, highhalf));
  } else {
    expx = __halves2bfloat162(lowhalf, __hneg(highhalf));
    expx = h2exp(expx);
    expx = __h2div(one, __hadd2(one, expx));
    lowhalf = __low2bfloat16(expx);
    highhalf = __high2bfloat16(expx);
    return __halves2bfloat162(__hsub(sone, lowhalf), highhalf);
  }
}

template <>
__device__ __forceinline__ __nv_bfloat162 logsigmoid_g(const __nv_bfloat162 x) {
  const __nv_bfloat162 zero = dscalar_zero<__nv_bfloat162>();
  const __nv_bfloat162 one = dscalar_one<__nv_bfloat162>();
  const __nv_bfloat16 szero = dscalar_zero<__nv_bfloat16>();
  const __nv_bfloat16 sone = dscalar_one<__nv_bfloat16>();
  __nv_bfloat162 expx;
  __nv_bfloat16 lowhalf = __low2bfloat16(x);
  __nv_bfloat16 highhalf = __high2bfloat16(x);
  // printf("Sigmoid: %f, %f\n", __nv_bfloat162float(lowhalf),
  // __nv_bfloat162float(highhalf));

  if (__hbgt2(x, zero)) {
    expx = h2exp(__hneg2(x));
    expx = __hneg2(h2log(__hadd2(one, expx)));
    return expx;
  } else if (__hblt2(x, zero)) {
    expx = h2exp(x);
    expx = __hsub2(x, h2log(__hadd2(one, expx)));
    return expx;
  } else if (__hgt(lowhalf, szero)) {
    expx = __halves2bfloat162(__hneg(lowhalf), highhalf);
    expx = h2exp(expx);
    expx = __hneg2(h2log(__hadd2(one, expx)));
    lowhalf = __low2bfloat16(expx);
    highhalf = __hadd(highhalf, __high2bfloat16(expx));
    return __halves2bfloat162(lowhalf, highhalf);
  } else {
    expx = __halves2bfloat162(lowhalf, __hneg(highhalf));
    expx = h2exp(expx);
    expx = __hneg2(h2log(__hadd2(one, expx)));
    lowhalf = __hadd(lowhalf, __low2bfloat16(expx));
    highhalf = __high2bfloat16(expx);
    return __halves2bfloat162(lowhalf, highhalf);
  }
}

template <>
__device__ __forceinline__ __nv_bfloat162
sigmoid_unstable_g(const __nv_bfloat162 x) {
  const __nv_bfloat162 one = dscalar_one<__nv_bfloat162>();
  return __h2div(one, __hadd2(one, h2exp(__hneg2(x))));
}

template <>
__device__ __forceinline__ __nv_bfloat162
d_sigmoid_g(const __nv_bfloat162 sigmoid_output) {
  return __hmul2(sigmoid_output,
                 __hsub2(dscalar_one<__nv_bfloat162>(), sigmoid_output));
}

template <>
__device__ __forceinline__ __nv_bfloat162
d_tanh_g(const __nv_bfloat162 tanh_output) {
  return __hsub2(dscalar_one<__nv_bfloat162>(),
                 __hmul2(tanh_output, tanh_output));
}

template <>
__device__ __forceinline__ __nv_bfloat162 max_g(const __nv_bfloat162 a,
                                                const __nv_bfloat162 b) {
  return __hmax2(a, b);
}

template <>
__device__ __forceinline__ __nv_bfloat16
low_half_2h<__nv_bfloat162, __nv_bfloat16>(const __nv_bfloat162 x) {
  return __low2bfloat16(x);
}

template <>
__device__ __forceinline__ __nv_bfloat16
high_half_2h<__nv_bfloat162, __nv_bfloat16>(const __nv_bfloat162 x) {
  return __high2bfloat16(x);
}

template <> __device__ __forceinline__ bool gt_zero_g(const __nv_bfloat162 x) {
  return __hbgt2(x, __float2bfloat162_rn(0.0f));
}

template <> __device__ __forceinline__ bool lt_zero_g(const __nv_bfloat162 x) {
  return __hblt2(x, __float2bfloat162_rn(0.0f));
}

template <>
__device__ __forceinline__ bool low_half_gt_zero_2h(const __nv_bfloat162 x) {
  return __hgt(__low2bfloat16(x), __float2bfloat16(0.0f));
}

template <>
__device__ __forceinline__ bool high_half_gt_zero_2h(const __nv_bfloat162 x) {
  return __hgt(__low2bfloat16(x), __float2bfloat16(0.0f));
}

template <>
__device__ __forceinline__ __nv_bfloat162
join_halves_2h(const __nv_bfloat16 a, const __nv_bfloat16 b) {
  return __halves2bfloat162(a, b);
}
