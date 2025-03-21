// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/kernels/bf16_math.cuh"
#include "mlx/backend/cuda/kernels/cucomplex_math.cuh"

#include <cuda_fp16.h>
#include <cuda/std/array>
#include <cuda/std/type_traits>

namespace mlx::core::mxcuda {

struct Add {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x + y;
  }
};

struct FloorDivide {
  template <typename T>
  __device__ T operator()(T x, T y) {
    if constexpr (
        cuda::std::is_same_v<T, float> || cuda::std::is_same_v<T, __half> ||
        cuda::std::is_same_v<T, __nv_bfloat16>) {
      return trunc(x / y);
    } else {
      return x / y;
    }
  }
};

struct Divide {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x / y;
  }
};

struct Remainder {
  template <typename T>
  __device__ cuda::std::
      enable_if_t<cuda::std::is_integral_v<T> & !cuda::std::is_signed_v<T>, T>
      operator()(T x, T y) {
    return x % y;
  }

  template <typename T>
  __device__ cuda::std::
      enable_if_t<cuda::std::is_integral_v<T> & cuda::std::is_signed_v<T>, T>
      operator()(T x, T y) {
    auto r = x % y;
    if (r != 0 && (r < 0 != y < 0)) {
      r += y;
    }
    return r;
  }

  template <typename T>
  __device__ cuda::std::enable_if_t<!cuda::std::is_integral_v<T>, T> operator()(
      T x,
      T y) {
    T r = fmod(x, y);
    if (r != 0 && (r < 0 != y < 0)) {
      r += y;
    }
    return r;
  }

  template <typename T>
  __device__ cuda::std::enable_if_t<cuda::std::is_same_v<T, cuComplex>>
  operator()(T x, T y) {
    return x % y;
  }
};

struct Equal {
  template <typename T>
  __device__ bool operator()(T x, T y) {
    return x == y;
  }
};

struct NaNEqual {
  template <typename T>
  __device__ bool operator()(T x, T y) {
    if constexpr (std::is_same_v<T, cuComplex>) {
      return x == y ||
          (isnan(x.real) && isnan(y.real) && isnan(x.imag) && isnan(y.imag)) ||
          (x.real == y.real && isnan(x.imag) && isnan(y.imag)) ||
          (isnan(x.real) && isnan(y.real) && x.imag == y.imag);
    } else {
      return x == y || (isnan(x) && isnan(y));
    }
  }
};

struct Greater {
  template <typename T>
  __device__ bool operator()(T x, T y) {
    return x > y;
  }
};

struct GreaterEqual {
  template <typename T>
  __device__ bool operator()(T x, T y) {
    return x >= y;
  }
};

struct Less {
  template <typename T>
  __device__ bool operator()(T x, T y) {
    return x < y;
  }
};

struct LessEqual {
  template <typename T>
  __device__ bool operator()(T x, T y) {
    return x <= y;
  }
};

struct LogAddExp {
  template <typename T>
  __device__ T operator()(T x, T y) {
    if (isnan(x) || isnan(y)) {
      return cuda::std::numeric_limits<T>::quiet_NaN();
    }
    constexpr T inf = cuda::std::numeric_limits<T>::infinity();
    T maxval = cuda::std::max(x, y);
    T minval = cuda::std::min(x, y);
    return (minval == -inf || maxval == inf)
        ? maxval
        : (maxval + log1p(expf(minval - maxval)));
  };
};

struct Maximum {
  template <typename T>
  __device__ cuda::std::enable_if_t<cuda::std::is_integral_v<T>, T> operator()(
      T x,
      T y) {
    return cuda::std::max(x, y);
  }

  template <typename T>
  __device__ cuda::std::enable_if_t<!cuda::std::is_integral_v<T>, T> operator()(
      T x,
      T y) {
    if (isnan(x)) {
      return x;
    }
    return x > y ? x : y;
  }

  template <typename T>
  __device__ cuda::std::enable_if_t<cuda::std::is_same_v<T, cuComplex>>
  operator()(T x, T y) {
    if (isnan(x.real) || isnan(x.imag)) {
      return x;
    }
    return x > y ? x : y;
  }
};

struct Minimum {
  template <typename T>
  __device__ cuda::std::enable_if_t<cuda::std::is_integral_v<T>, T> operator()(
      T x,
      T y) {
    return cuda::std::min(x, y);
  }

  template <typename T>
  __device__ cuda::std::enable_if_t<!cuda::std::is_integral_v<T>, T> operator()(
      T x,
      T y) {
    if (isnan(x)) {
      return x;
    }
    return x < y ? x : y;
  }

  template <typename T>
  __device__ cuda::std::enable_if_t<cuda::std::is_same_v<T, cuComplex>>
  operator()(T x, T y) {
    if (isnan(x.real) || isnan(x.imag)) {
      return x;
    }
    return x < y ? x : y;
  }
};

struct Multiply {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x * y;
  }
};

struct NotEqual {
  template <typename T>
  __device__ bool operator()(T x, T y) {
    if constexpr (std::is_same_v<T, cuComplex>) {
      return x.real != y.real || x.imag != y.imag;
    } else {
      return x != y;
    }
  }
};

struct Power {
  template <typename T>
  __device__ cuda::std::enable_if_t<
      !cuda::std::is_integral_v<T> && !cuda::std::is_same_v<T, cuComplex>,
      T>
  operator()(T base, T exp) {
    return powf(base, exp);
  }

  template <typename T>
  __device__ cuda::std::enable_if_t<cuda::std::is_integral_v<T>, T> operator()(
      T base,
      T exp) {
    T res = 1;
    while (exp) {
      if (exp & 1) {
        res *= base;
      }
      exp >>= 1;
      base *= base;
    }
    return res;
  }

  template <typename T>
  __device__ cuda::std::enable_if_t<cuda::std::is_same_v<T, cuComplex>>
  operator()(T x, T y) {
    auto x_theta = atan2f(x.imag, x.real);
    auto x_ln_r = 0.5 * logf(x.real * x.real + x.imag * x.imag);
    auto mag = expf(y.real * x_ln_r - y.imag * x_theta);
    auto phase = y.imag * x_ln_r + y.real * x_theta;
    return {mag * cosf(phase), mag * sinf(phase)};
  }
};

struct Subtract {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x - y;
  }
};

struct LogicalAnd {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x && y;
  };
};

struct LogicalOr {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x || y;
  };
};

struct BitwiseAnd {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x & y;
  };
};

struct BitwiseOr {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x | y;
  };
};

struct BitwiseXor {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x ^ y;
  };
};

struct LeftShift {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x << y;
  };
};

struct RightShift {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x >> y;
  };
};

struct ArcTan2 {
  template <typename T>
  __device__ T operator()(T y, T x) {
    return atan2f(y, x);
  }
};

struct DivMod {
  template <typename T>
  __device__ cuda::std::array<T, 2> operator()(T x, T y) {
    return {FloorDivide{}(x, y), Remainder{}(x, y)};
  };
};

} // namespace mlx::core::mxcuda
