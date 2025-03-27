// Copyright © 2025 Apple Inc.

#pragma once

#include <cuComplex.h>
#include <cuda_fp16.h>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

namespace mlx::core::mxcuda {

///////////////////////////////////////////////////////////////////////////////
// Constant values for half types.
///////////////////////////////////////////////////////////////////////////////

#define MLX_DEFINE_CONSTEXPR_VALUE(NAME, HALF_VALUE, BF16_VALUE, ...) \
  template <typename T>                                               \
  constexpr __host__ __device__ T NAME() {                            \
    if constexpr (cuda::std::is_same_v<T, __half>) {                  \
      uint16_t value = HALF_VALUE;                                    \
      return __builtin_bit_cast(__half, value);                       \
    } else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>) {    \
      uint16_t value = BF16_VALUE;                                    \
      return __builtin_bit_cast(__nv_bfloat16, value);                \
    } else {                                                          \
      __VA_ARGS__                                                     \
    }                                                                 \
  }

MLX_DEFINE_CONSTEXPR_VALUE(zero_value, 0x0000, 0x0000, {
  if constexpr (cuda::std::is_same_v<T, cuComplex>) {
    return cuComplex{0, 0};
  } else {
    return 0;
  }
})

MLX_DEFINE_CONSTEXPR_VALUE(one_value, 0x3C00, 0x3F80, {
  if constexpr (cuda::std::is_same_v<T, cuComplex>) {
    return cuComplex{1, 1};
  } else {
    return 1;
  }
})

MLX_DEFINE_CONSTEXPR_VALUE(infinite_value, 0x7C00, 0x7F80, {
  return cuda::std::numeric_limits<T>::infinity();
})

MLX_DEFINE_CONSTEXPR_VALUE(negative_infinite_value, 0xFC00, 0xFF80, {
  return -cuda::std::numeric_limits<T>::infinity();
})

MLX_DEFINE_CONSTEXPR_VALUE(finite_max_value, 0x7BFF, 0x7F7F, {
  return cuda::std::numeric_limits<T>::max();
})

MLX_DEFINE_CONSTEXPR_VALUE(finite_min_value, 0xFBFF, 0xFF7F, {
  return cuda::std::numeric_limits<T>::min();
})

#undef MLX_DEFINE_CONSTEXPR_VALUE

///////////////////////////////////////////////////////////////////////////////
// Unary ops for half types.
///////////////////////////////////////////////////////////////////////////////

#if __CUDA_ARCH__ >= 800
#define MLX_DEFINE_UNARY_OP(NAME, HALF_OP)                         \
  template <typename T>                                            \
  __forceinline__ __device__ auto NAME(T x) {                      \
    if constexpr (cuda::std::is_same_v<T, __half>) {               \
      return HALF_OP(x);                                           \
    } else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>) { \
      return HALF_OP(x);                                           \
    } else {                                                       \
      return ::NAME(x);                                            \
    }                                                              \
  }
#else
#define MLX_DEFINE_UNARY_OP(NAME, HALF_OP)           \
  template <typename T>                              \
  __forceinline__ __device__ auto NAME(T x) {        \
    if constexpr (cuda::std::is_same_v<T, __half>) { \
      return HALF_OP(x);                             \
    } else {                                         \
      return ::NAME(x);                              \
    }                                                \
  }
#endif

MLX_DEFINE_UNARY_OP(abs, __habs)
MLX_DEFINE_UNARY_OP(isnan, __hisnan)
MLX_DEFINE_UNARY_OP(log, hlog)
MLX_DEFINE_UNARY_OP(log2, hlog2)
MLX_DEFINE_UNARY_OP(log10, hlog10)
MLX_DEFINE_UNARY_OP(log1p, hlog1p)
MLX_DEFINE_UNARY_OP(rint, hrint)

#undef MLX_DEFINE_UNARY_OP

///////////////////////////////////////////////////////////////////////////////
// Binary ops for half types.
///////////////////////////////////////////////////////////////////////////////

#if __CUDA_ARCH__ >= 800
#define MLX_DEFINE_BINARY_OP(NAME, HALF_OP)                        \
  template <typename T>                                            \
  __forceinline__ __device__ auto NAME(T x, T y) {                 \
    if constexpr (cuda::std::is_same_v<T, __half>) {               \
      return HALF_OP(x, y);                                        \
    } else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>) { \
      return HALF_OP(x, y);                                        \
    } else {                                                       \
      return ::NAME(x, y);                                         \
    }                                                              \
  }
#else
#define MLX_DEFINE_BINARY_OP(NAME, HALF_OP)          \
  template <typename T>                              \
  __forceinline__ __device__ auto NAME(T x, T y) {   \
    if constexpr (cuda::std::is_same_v<T, __half>) { \
      return HALF_OP(x, y);                          \
    } else {                                         \
      return ::NAME(x, y);                           \
    }                                                \
  }
#endif

MLX_DEFINE_BINARY_OP(max, __hmax)
MLX_DEFINE_BINARY_OP(min, __hmin)

#undef MLX_DEFINE_BINARY_OP

template <typename T>
__forceinline__ __device__ T fmod(T x, T y) {
  if constexpr (cuda::std::is_same_v<T, __half>) {
    return __float2half(::fmod(__half2float(x), __half2float(y)));
#if __CUDA_ARCH__ >= 800
  } else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>) {
    return __float2bfloat16(::fmod(__bfloat162float(x), __bfloat162float(y)));
#endif
  } else {
    return ::fmod(x, y);
  }
}

///////////////////////////////////////////////////////////////////////////////
// Missing C++ operator overrides for CUDA 7.
///////////////////////////////////////////////////////////////////////////////

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800

#define MLX_DEFINE_BF16_OP(OP)                                           \
  __forceinline__ __device__ __nv_bfloat16 operator OP(                  \
      __nv_bfloat16 x, __nv_bfloat16 y) {                                \
    return __float2bfloat16(__bfloat162float(x) OP __bfloat162float(y)); \
  }

#define MLX_DEFINE_BF16_CMP(OP)                                          \
  __forceinline__ __device__ bool operator OP(                           \
      __nv_bfloat16 x, __nv_bfloat16 y) {                                \
    return __float2bfloat16(__bfloat162float(x) OP __bfloat162float(y)); \
  }

MLX_DEFINE_BF16_OP(+)
MLX_DEFINE_BF16_OP(-)
MLX_DEFINE_BF16_OP(*)
MLX_DEFINE_BF16_OP(/)
MLX_DEFINE_BF16_CMP(>)
MLX_DEFINE_BF16_CMP(<)
MLX_DEFINE_BF16_CMP(>=)
MLX_DEFINE_BF16_CMP(<=)

#undef MLX_DEFINE_BF16_OP
#undef MLX_DEFINE_BF16_CMP

#endif // __CUDA_ARCH__ < 800

///////////////////////////////////////////////////////////////////////////////
// Additional C++ operator overrides between half types and native types.
///////////////////////////////////////////////////////////////////////////////

#define MLX_DEFINE_HALF_OP(HALF, HALF2FLOAT, FLOAT2HALF, OP)             \
  template <                                                             \
      typename T,                                                        \
      typename = cuda::std::enable_if_t<!cuda::std::is_same_v<T, HALF>>> \
  __forceinline__ __device__ HALF operator OP(HALF x, T y) {             \
    return FLOAT2HALF(HALF2FLOAT(x) OP static_cast<float>(y));           \
  }                                                                      \
  template <                                                             \
      typename T,                                                        \
      typename = cuda::std::enable_if_t<!cuda::std::is_same_v<T, HALF>>> \
  __forceinline__ __device__ bool operator OP(T x, HALF y) {             \
    return FLOAT2HALF(static_cast<float>(x) OP HALF2FLOAT(y));           \
  }

#define MLX_DEFINE_HALF_CMP(HALF, HALF2FLOAT, OP)                        \
  template <                                                             \
      typename T,                                                        \
      typename = cuda::std::enable_if_t<!cuda::std::is_same_v<T, HALF>>> \
  __forceinline__ __device__ bool operator OP(HALF x, T y) {             \
    return HALF2FLOAT(x) OP static_cast<float>(y);                       \
  }                                                                      \
  template <                                                             \
      typename T,                                                        \
      typename = cuda::std::enable_if_t<!cuda::std::is_same_v<T, HALF>>> \
  __forceinline__ __device__ bool operator OP(T x, HALF y) {             \
    return static_cast<float>(y) OP HALF2FLOAT(x);                       \
  }

MLX_DEFINE_HALF_OP(__half, __half2float, __float2half, +)
MLX_DEFINE_HALF_OP(__half, __half2float, __float2half, -)
MLX_DEFINE_HALF_OP(__half, __half2float, __float2half, *)
MLX_DEFINE_HALF_OP(__half, __half2float, __float2half, /)
MLX_DEFINE_HALF_OP(__nv_bfloat16, __bfloat162float, __float2bfloat16, +)
MLX_DEFINE_HALF_OP(__nv_bfloat16, __bfloat162float, __float2bfloat16, -)
MLX_DEFINE_HALF_OP(__nv_bfloat16, __bfloat162float, __float2bfloat16, *)
MLX_DEFINE_HALF_OP(__nv_bfloat16, __bfloat162float, __float2bfloat16, /)
MLX_DEFINE_HALF_CMP(__half, __half2float, <)
MLX_DEFINE_HALF_CMP(__half, __half2float, >)
MLX_DEFINE_HALF_CMP(__half, __half2float, <=)
MLX_DEFINE_HALF_CMP(__half, __half2float, >=)
MLX_DEFINE_HALF_CMP(__half, __half2float, ==)
MLX_DEFINE_HALF_CMP(__half, __half2float, !=)
MLX_DEFINE_HALF_CMP(__nv_bfloat16, __bfloat162float, <)
MLX_DEFINE_HALF_CMP(__nv_bfloat16, __bfloat162float, >)
MLX_DEFINE_HALF_CMP(__nv_bfloat16, __bfloat162float, <=)
MLX_DEFINE_HALF_CMP(__nv_bfloat16, __bfloat162float, >=)
MLX_DEFINE_HALF_CMP(__nv_bfloat16, __bfloat162float, ==)
MLX_DEFINE_HALF_CMP(__nv_bfloat16, __bfloat162float, !=)

#undef MLX_DEFINE_HALF_OP
#undef MLX_DEFINE_HALF_CMP

} // namespace mlx::core::mxcuda
