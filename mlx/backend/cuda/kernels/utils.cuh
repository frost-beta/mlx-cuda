// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/kernels/fp16_math.cuh"

#include <cuComplex.h>
#include <cuda/std/array>
#include <cuda/std/limits>

namespace mlx::core::cu {

///////////////////////////////////////////////////////////////////////////////
// CUDA kernel utils
///////////////////////////////////////////////////////////////////////////////

// All existing NVIDIA hardware has a fixed 32 warp size. Though a built-in
// warpSize variable exists, using it would prevent compile-time optimizations.
#define WARP_SIZE 32

// Helper macros for dispatch macros (see below).
#define MLX_INTERNAL_IF_CASE(DIM, BLOCK_DIM, ...) \
  }                                               \
  else if (_num_threads <= DIM) {                 \
    constexpr uint32_t BLOCK_DIM = DIM;           \
    __VA_ARGS__;

#define MLX_INTERNAL_IF_CASE_DIMS(NUM_THREADS, BLOCK_DIM, ...) \
  {                                                            \
    uint32_t _num_threads = NUM_THREADS;                       \
    if (false) {                                               \
      MLX_INTERNAL_IF_CASE(32, BLOCK_DIM, __VA_ARGS__)         \
      MLX_INTERNAL_IF_CASE(64, BLOCK_DIM, __VA_ARGS__)         \
      MLX_INTERNAL_IF_CASE(128, BLOCK_DIM, __VA_ARGS__)        \
      MLX_INTERNAL_IF_CASE(256, BLOCK_DIM, __VA_ARGS__)        \
      MLX_INTERNAL_IF_CASE(512, BLOCK_DIM, __VA_ARGS__)        \
    } else {                                                   \
      constexpr uint32_t BLOCK_DIM = 1024;                     \
      __VA_ARGS__;                                             \
    }                                                          \
  }

// Some kernels use CUB which requires block_dim to be known at compile-time,
// use this macro to dispatch constexpr block_dim for the num_threads.
#define MLX_SWITCH_BLOCK_DIM(NUM_THREADS, BLOCK_DIM, ...) \
  MLX_INTERNAL_IF_CASE_DIMS(NUM_THREADS, BLOCK_DIM, __VA_ARGS__)

// To pass shape/strides to kernels via constant memory, their size must be
// known at compile time.
#define MAX_NDIM 8

using Shape = cuda::std::array<int32_t, MAX_NDIM>;
using Strides = cuda::std::array<int64_t, MAX_NDIM>;

// Utility to copy data from vector to array in host.
template <typename T>
inline cuda::std::array<T, MAX_NDIM> const_param(const std::vector<T>& vec) {
  if (vec.size() > MAX_NDIM) {
    throw std::runtime_error("ndim can not be larger than 8.");
  }
  cuda::std::array<T, MAX_NDIM> result;
  std::copy_n(vec.begin(), vec.size(), result.begin());
  return result;
}

///////////////////////////////////////////////////////////////////////////////
// Type limits utils
///////////////////////////////////////////////////////////////////////////////

template <typename U>
struct Limits {
  static constexpr U max = cuda::std::numeric_limits<U>::max();
  static constexpr U min = cuda::std::numeric_limits<U>::min();
  static constexpr U finite_max = cuda::std::numeric_limits<U>::max();
  static constexpr U finite_min = cuda::std::numeric_limits<U>::min();
};

template <>
struct Limits<bool> {
  static constexpr bool max = true;
  static constexpr bool min = false;
};

template <>
struct Limits<cuComplex> {
  static constexpr cuComplex max = {
      cuda::std::numeric_limits<float>::infinity(),
      cuda::std::numeric_limits<float>::infinity()};
  static constexpr cuComplex min = {
      -cuda::std::numeric_limits<float>::infinity(),
      -cuda::std::numeric_limits<float>::infinity()};
};

// Like MLX_FORALL_FLOAT_TYPES but use CUDA types.
#define MLX_FORALL_CUDA_FLOAT_TYPES(_) \
  _(float, float32)                    \
  _(double, float64)                   \
  _(__half, float16)                   \
  _(__nv_bfloat16, bfloat16)

// Some CCCL/CUDA combinations do not provide constexpr limits for half types.
#define SPECIALIZE_FloatLimits(CPP_TYPE, DTYPE)                          \
  template <>                                                            \
  struct Limits<CPP_TYPE> {                                              \
    static constexpr CPP_TYPE max = infinite_value<CPP_TYPE>();          \
    static constexpr CPP_TYPE min = negative_infinite_value<CPP_TYPE>(); \
    static constexpr CPP_TYPE finite_max = max_value<CPP_TYPE>();        \
    static constexpr CPP_TYPE finite_min = lowest_value<CPP_TYPE>();     \
  };

MLX_FORALL_CUDA_FLOAT_TYPES(SPECIALIZE_FloatLimits)

#undef SPECIALIZE_FloatLimits

///////////////////////////////////////////////////////////////////////////////
// Indexing utils
///////////////////////////////////////////////////////////////////////////////

template <typename IdxT = int64_t>
inline __host__ __device__ IdxT
elem_to_loc(IdxT elem, const int* shape, const int64_t* strides, int ndim) {
  IdxT loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += (elem % shape[i]) * IdxT(strides[i]);
    elem /= shape[i];
  }
  return loc;
}

} // namespace mlx::core::cu
