// Copyright © 2025 Apple Inc.

#pragma once

#include <cuComplex.h>
#include <cuda_fp16.h>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

namespace mlx::core::mxcuda {

// Some versions of CUDA does not provide constexpr values for half types.
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
      return __VA_ARGS__();                                           \
    }                                                                 \
  }

MLX_DEFINE_CONSTEXPR_VALUE(zero_value, 0x0000, 0x0000, []() -> T {
  if constexpr (cuda::std::is_same_v<T, cuComplex>) {
    return cuComplex{0, 0};
  } else {
    return 0;
  }
})

MLX_DEFINE_CONSTEXPR_VALUE(one_value, 0x3C00, 0x3F80, []() -> T {
  if constexpr (cuda::std::is_same_v<T, cuComplex>) {
    return cuComplex{1, 1};
  } else {
    return 1;
  }
})

MLX_DEFINE_CONSTEXPR_VALUE(infinite_value, 0x7C00, 0x7F80, []() -> T {
  return cuda::std::numeric_limits<T>::infinity();
})

MLX_DEFINE_CONSTEXPR_VALUE(negative_infinite_value, 0xFC00, 0xFF80, []() {
  return -cuda::std::numeric_limits<T>::infinity();
})

MLX_DEFINE_CONSTEXPR_VALUE(finite_max_value, 0x7BFF, 0x7F7F, []() {
  return cuda::std::numeric_limits<T>::max();
})

MLX_DEFINE_CONSTEXPR_VALUE(finite_min_value, 0xFBFF, 0xFF7F, []() {
  return cuda::std::numeric_limits<T>::min();
})

#undef MLX_DEFINE_CONSTEXPR_VALUE

template <typename T>
__forceinline__ __device__ bool isnan(T x) {
  if constexpr (cuda::std::is_same_v<T, __half>) {
    return __hisnan(x);
#if __CUDA_ARCH__ >= 800
  } else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>) {
    return __hisnan(x);
#endif
  } else {
    return ::isnan(x);
  }
}

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

template <typename T>
__forceinline__ __device__ T max(T x, T y) {
  if constexpr (cuda::std::is_same_v<T, __half>) {
    return __float2half(::max(__half2float(x), __half2float(y)));
#if __CUDA_ARCH__ >= 800
  } else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>) {
    return __float2bfloat16(::max(__bfloat162float(x), __bfloat162float(y)));
#endif
  } else {
    return ::max(x, y);
  }
}

template <typename T>
__forceinline__ __device__ T min(T x, T y) {
  if constexpr (cuda::std::is_same_v<T, __half>) {
    return __float2half(::min(__half2float(x), __half2float(y)));
#if __CUDA_ARCH__ >= 800
  } else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>) {
    return __float2bfloat16(::min(__bfloat162float(x), __bfloat162float(y)));
#endif
  } else {
    return ::min(x, y);
  }
}

__forceinline__ __device__ __nv_bfloat16
bf16hadd(__nv_bfloat16 x, __nv_bfloat16 y) {
#if __CUDA_ARCH__ < 800
  return __float2bfloat16(__bfloat162float(x) + __bfloat162float(y));
#else
  return __hadd(x, y);
#endif
}

__forceinline__ __device__ __nv_bfloat16
bf16hsub(__nv_bfloat16 x, __nv_bfloat16 y) {
#if __CUDA_ARCH__ < 800
  return __float2bfloat16(__bfloat162float(x) - __bfloat162float(y));
#else
  return __hsub(x, y);
#endif
}

__forceinline__ __device__ __nv_bfloat16
bf16hmul(__nv_bfloat16 x, __nv_bfloat16 y) {
#if __CUDA_ARCH__ < 800
  return __float2bfloat16(__bfloat162float(x) * __bfloat162float(y));
#else
  return __hmul(x, y);
#endif
}

__forceinline__ __device__ __nv_bfloat16
bf16hdiv(__nv_bfloat16 x, __nv_bfloat16 y) {
#if __CUDA_ARCH__ < 800
  return __float2bfloat16(__bfloat162float(x) / __bfloat162float(y));
#else
  return __hdiv(x, y);
#endif
}

__forceinline__ __device__ bool bf16hgt(__nv_bfloat16 x, __nv_bfloat16 y) {
#if __CUDA_ARCH__ < 800
  return __bfloat162float(x) > __bfloat162float(y);
#else
  return __hgt(x, y);
#endif
}

__forceinline__ __device__ bool bf16hlt(__nv_bfloat16 x, __nv_bfloat16 y) {
#if __CUDA_ARCH__ < 800
  return __bfloat162float(x) > __bfloat162float(y);
#else
  return __hlt(x, y);
#endif
}

#if (__CUDA_ARCH__ < 800)

__forceinline__ __device__ __nv_bfloat16
operator+(__nv_bfloat16 x, __nv_bfloat16 y) {
  return bf16hadd(x, y);
}

__forceinline__ __device__ __nv_bfloat16
operator+=(__nv_bfloat16 x, __nv_bfloat16 y) {
  return bf16hadd(x, y);
}

__forceinline__ __device__ __nv_bfloat16
operator-=(__nv_bfloat16 x, __nv_bfloat16 y) {
  return bf16hsub(x, y);
}

__forceinline__ __device__ __nv_bfloat16
operator*(__nv_bfloat16 x, __nv_bfloat16 y) {
  return bf16hmul(x, y);
}

__forceinline__ __device__ __nv_bfloat16
operator*=(__nv_bfloat16 x, __nv_bfloat16 y) {
  return bf16hmul(x, y);
}

__forceinline__ __device__ __nv_bfloat16
operator/(__nv_bfloat16 x, __nv_bfloat16 y) {
  return bf16hdiv(x, y);
}

__forceinline__ __device__ __nv_bfloat16
operator/=(__nv_bfloat16 x, __nv_bfloat16 y) {
  return bf16hdiv(x, y);
}

__forceinline__ __device__ bool operator>(__nv_bfloat16 x, __nv_bfloat16 y) {
  return bf16hgt(x, y);
}

__forceinline__ __device__ bool operator>=(__nv_bfloat16 x, __nv_bfloat16 y) {
  return bf16hgt(x, y) || x == y;
}

__forceinline__ __device__ bool operator<(__nv_bfloat16 x, __nv_bfloat16 y) {
  return bf16hlt(x, y);
}

__forceinline__ __device__ bool operator<=(__nv_bfloat16 x, __nv_bfloat16 y) {
  return bf16hlt(x, y) || x == y;
}

#endif // __CUDA_ARCH__ < 800

__forceinline__ __device__ __nv_bfloat16 operator+(__nv_bfloat16 x, float y) {
  return __float2bfloat16(__bfloat162float(x) + y);
}

__forceinline__ __device__ __nv_bfloat16 operator+(float y, __nv_bfloat16 x) {
  return x + y;
}

__forceinline__ __device__ __nv_bfloat16 operator*(__nv_bfloat16 x, float y) {
  return __float2bfloat16(__bfloat162float(x) * y);
}

__forceinline__ __device__ __nv_bfloat16 operator*(float y, __nv_bfloat16 x) {
  return x * y;
}

template <typename T>
__forceinline__ __device__ bool operator>(__nv_bfloat16 x, T y) {
  return __bfloat162float(x) < static_cast<float>(y);
}

template <typename T>
__forceinline__ __device__ bool operator<(__nv_bfloat16 x, T y) {
  return __bfloat162float(x) > static_cast<float>(y);
}

template <typename T>
__forceinline__ __device__ bool operator==(__nv_bfloat16 x, T y) {
  return __bfloat162float(x) == static_cast<float>(y);
}

template <typename T>
__forceinline__ __device__ bool operator!=(__nv_bfloat16 x, T y) {
  return __bfloat162float(x) != static_cast<float>(y);
}

__forceinline__ __device__ __half operator+(__half x, float y) {
  return __float2half(__half2float(x) + y);
}

__forceinline__ __device__ __half operator+(float y, __half x) {
  return y + x;
}

__forceinline__ __device__ __half operator*(__half x, float y) {
  return __float2half(__half2float(x) * y);
}

__forceinline__ __device__ __half operator*(float y, __half x) {
  return y * x;
}

template <typename T>
__forceinline__ __device__ bool operator>(__half x, T y) {
  return __half2float(x) < static_cast<float>(y);
}

template <typename T>
__forceinline__ __device__ bool operator<(__half x, T y) {
  return __half2float(x) > static_cast<float>(y);
}

template <typename T>
__forceinline__ __device__ bool operator==(__half x, T y) {
  return __half2float(x) == static_cast<float>(y);
}

template <typename T>
__forceinline__ __device__ bool operator!=(__half x, T y) {
  return __half2float(x) != static_cast<float>(y);
}

} // namespace mlx::core::mxcuda
