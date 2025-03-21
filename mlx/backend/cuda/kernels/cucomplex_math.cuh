// Copyright © 2025 Apple Inc.
// Copyright © 2017-2024 The Simons Foundation, Inc.
//
// FINUFFT is licensed under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance with the
// License.  You may obtain a copy of the License at
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
// https://github.com/flatironinstitute/finufft/blob/main/include/cufinufft/contrib/helper_math.h

#pragma once

#include <cuComplex.h>

// This header provides some helper functions for cuComplex types.
// It mainly wraps existing CUDA implementations to provide operator overloads
// e.g. cuAdd, cuSub, cuMul, cuDiv, cuCreal, cuCimag, cuCabs, cuCarg, cuConj are
// all provided by CUDA

__forceinline__ __host__ __device__ cuDoubleComplex
operator+(const cuDoubleComplex& a, const cuDoubleComplex& b) {
  return cuCadd(a, b);
}

__forceinline__ __host__ __device__ cuDoubleComplex
operator-(const cuDoubleComplex& a, const cuDoubleComplex& b) {
  return cuCsub(a, b);
}

__forceinline__ __host__ __device__ cuDoubleComplex
operator*(const cuDoubleComplex& a, const cuDoubleComplex& b) {
  return cuCmul(a, b);
}

__forceinline__ __host__ __device__ cuDoubleComplex
operator/(const cuDoubleComplex& a, const cuDoubleComplex& b) {
  return cuCdiv(a, b);
}

__forceinline__ __host__ __device__ bool operator==(
    const cuDoubleComplex& a,
    const cuDoubleComplex& b) {
  return cuCreal(a) == cuCreal(b) && cuCimag(a) == cuCimag(b);
}

__forceinline__ __host__ __device__ bool operator!=(
    const cuDoubleComplex& a,
    const cuDoubleComplex& b) {
  return !(a == b);
}

__forceinline__ __host__ __device__ cuDoubleComplex
operator+(const cuDoubleComplex& a, double b) {
  return make_cuDoubleComplex(cuCreal(a) + b, cuCimag(a));
}

__forceinline__ __host__ __device__ cuDoubleComplex
operator+(double a, const cuDoubleComplex& b) {
  return make_cuDoubleComplex(a + cuCreal(b), cuCimag(b));
}

__forceinline__ __host__ __device__ cuDoubleComplex
operator-(const cuDoubleComplex& a, double b) {
  return make_cuDoubleComplex(cuCreal(a) - b, cuCimag(a));
}

__forceinline__ __host__ __device__ cuDoubleComplex
operator-(double a, const cuDoubleComplex& b) {
  return make_cuDoubleComplex(a - cuCreal(b), -cuCimag(b));
}

__forceinline__ __host__ __device__ cuDoubleComplex
operator*(const cuDoubleComplex& a, double b) {
  return make_cuDoubleComplex(cuCreal(a) * b, cuCimag(a) * b);
}

__forceinline__ __host__ __device__ cuDoubleComplex
operator*(double a, const cuDoubleComplex& b) {
  return make_cuDoubleComplex(a * cuCreal(b), a * cuCimag(b));
}

__forceinline__ __host__ __device__ cuDoubleComplex
operator/(const cuDoubleComplex& a, double b) {
  return make_cuDoubleComplex(cuCreal(a) / b, cuCimag(a) / b);
}

__forceinline__ __host__ __device__ cuDoubleComplex
operator/(double a, const cuDoubleComplex& b) {
  double denom = cuCreal(b) * cuCreal(b) + cuCimag(b) * cuCimag(b);
  return make_cuDoubleComplex(
      (a * cuCreal(b)) / denom, (-a * cuCimag(b)) / denom);
}

__forceinline__ __host__ __device__ cuFloatComplex
operator+(const cuFloatComplex& a, const cuFloatComplex& b) {
  return cuCaddf(a, b);
}

__forceinline__ __host__ __device__ cuFloatComplex
operator-(const cuFloatComplex& a, const cuFloatComplex& b) {
  return cuCsubf(a, b);
}

__forceinline__ __host__ __device__ cuFloatComplex
operator*(const cuFloatComplex& a, const cuFloatComplex& b) {
  return cuCmulf(a, b);
}

__forceinline__ __host__ __device__ cuFloatComplex
operator/(const cuFloatComplex& a, const cuFloatComplex& b) {
  return cuCdivf(a, b);
}

__forceinline__ __host__ __device__ bool operator==(
    const cuFloatComplex& a,
    const cuFloatComplex& b) {
  return cuCrealf(a) == cuCrealf(b) && cuCimagf(a) == cuCimagf(b);
}

__forceinline__ __host__ __device__ bool operator!=(
    const cuFloatComplex& a,
    const cuFloatComplex& b) {
  return !(a == b);
}

__forceinline__ __host__ __device__ cuFloatComplex
operator+(const cuFloatComplex& a, float b) {
  return make_cuFloatComplex(cuCrealf(a) + b, cuCimagf(a));
}

__forceinline__ __host__ __device__ cuFloatComplex
operator+(float a, const cuFloatComplex& b) {
  return make_cuFloatComplex(a + cuCrealf(b), cuCimagf(b));
}

__forceinline__ __host__ __device__ cuFloatComplex
operator-(const cuFloatComplex& a, float b) {
  return make_cuFloatComplex(cuCrealf(a) - b, cuCimagf(a));
}

__forceinline__ __host__ __device__ cuFloatComplex
operator-(float a, const cuFloatComplex& b) {
  return make_cuFloatComplex(a - cuCrealf(b), -cuCimagf(b));
}

__forceinline__ __host__ __device__ cuFloatComplex
operator*(const cuFloatComplex& a, float b) {
  return make_cuFloatComplex(cuCrealf(a) * b, cuCimagf(a) * b);
}

__forceinline__ __host__ __device__ cuFloatComplex
operator*(float a, const cuFloatComplex& b) {
  return make_cuFloatComplex(a * cuCrealf(b), a * cuCimagf(b));
}

__forceinline__ __host__ __device__ cuFloatComplex
operator/(const cuFloatComplex& a, float b) {
  return make_cuFloatComplex(cuCrealf(a) / b, cuCimagf(a) / b);
}

__forceinline__ __host__ __device__ cuFloatComplex
operator/(float a, const cuFloatComplex& b) {
  float denom = cuCrealf(b) * cuCrealf(b) + cuCimagf(b) * cuCimagf(b);
  return make_cuFloatComplex(
      (a * cuCrealf(b)) / denom, (-a * cuCimagf(b)) / denom);
}
