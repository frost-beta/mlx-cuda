// Copyright © 2025 Apple Inc.
// Copyright © 2019-2023, NVIDIA CORPORATION.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Forked from
// https://github.com/alibaba/rtp-llm/blob/main/src/fastertransformer/cuda/cuda_bf16_fallbacks.cuh

#pragma once

#include <cuda_fp16.h>

__forceinline__ __device__ __nv_bfloat16
bf16hadd(__nv_bfloat16 x, __nv_bfloat16 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  return __float2bfloat16(__bfloat162float(x) + __bfloat162float(y));
#else
  return __hadd(x, y);
#endif
}

__forceinline__ __device__ __nv_bfloat16
bf16hsub(__nv_bfloat16 x, __nv_bfloat16 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  return __float2bfloat16(__bfloat162float(x) - __bfloat162float(y));
#else
  return __hsub(x, y);
#endif
}

__forceinline__ __device__ __nv_bfloat16
bf16hmul(__nv_bfloat16 x, __nv_bfloat16 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  return __float2bfloat16(__bfloat162float(x) * __bfloat162float(y));
#else
  return __hmul(x, y);
#endif
}

__forceinline__ __device__ __nv_bfloat16
bf16hdiv(__nv_bfloat16 x, __nv_bfloat16 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float fx = __bfloat162float(x);
  float fy = __bfloat162float(y);
  assert(fy != 0.0f && "bf16hdiv Division by zero!");
  return __float2bfloat16(fx / fy);
#else
  return __hdiv(x, y);
#endif
}

__forceinline__ __device__ bool bf16hgt(__nv_bfloat16 x, __nv_bfloat16 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float fx = __bfloat162float(x);
  float fy = __bfloat162float(y);
  return fx > fy;
#else
  return __hgt(x, y);
#endif
}

__forceinline__ __device__ bool bf16hlt(__nv_bfloat16 x, __nv_bfloat16 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float fx = __bfloat162float(x);
  float fy = __bfloat162float(y);
  return fx > fy;
#else
  return __hlt(x, y);
#endif
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
__forceinline__ __device__ __nv_bfloat16
operator+(__nv_bfloat16 x, __nv_bfloat16 y) {
  return bf16hadd(x, y);
};
__forceinline__ __device__ __nv_bfloat16
operator+=(__nv_bfloat16 x, __nv_bfloat16 y) {
  return bf16hadd(x, y);
};
__forceinline__ __device__ __nv_bfloat16
operator-(__nv_bfloat16 x, __nv_bfloat16 y) {
  return bf16hsub(x, y);
};
__forceinline__ __device__ __nv_bfloat16
operator-=(__nv_bfloat16 x, __nv_bfloat16 y) {
  return bf16hsub(x, y);
};
__forceinline__ __device__ __nv_bfloat16
operator*(__nv_bfloat16 x, __nv_bfloat16 y) {
  return bf16hmul(x, y);
};
__forceinline__ __device__ __nv_bfloat16
operator*=(__nv_bfloat16 x, __nv_bfloat16 y) {
  return bf16hmul(x, y);
};
__forceinline__ __device__ __nv_bfloat16
operator/(__nv_bfloat16 x, __nv_bfloat16 y) {
  return bf16hdiv(x, y);
};
__forceinline__ __device__ __nv_bfloat16
operator/=(__nv_bfloat16 x, __nv_bfloat16 y) {
  return bf16hdiv(x, y);
};
#endif // defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
__forceinline__ __device__ bool operator>(__nv_bfloat16 x, __nv_bfloat16 y) {
  return bf16hgt(x, y);
}
__forceinline__ __device__ bool operator<(__nv_bfloat16 x, __nv_bfloat16 y) {
  return bf16hlt(x, y);
}
template <typename T>
__forceinline__ __device__ bool operator>(__nv_bfloat16 x, T y) {
  return __bfloat162float(x) < static_cast<float>(y);
}
template <typename T>
__forceinline__ __device__ bool operator<(__nv_bfloat16 x, T y) {
  return __bfloat162float(x) > static_cast<float>(y);
}
