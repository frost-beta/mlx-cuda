// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/kernels/utils.cuh"

namespace mlx::core::mxcuda {

struct Abs {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_unsigned_v<T>) {
      return x;
    } else if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      return {sqrt(cuCrealf(x) * cuCrealf(x) + cuCimagf(x) * cuCimagf(x)), 0};
    } else {
      return abs(x);
    }
  }
};

struct Log {
  template <typename T>
  __device__ T operator()(T x) {
    return log(x);
  }
};

struct Log2 {
  template <typename T>
  __device__ T operator()(T x) {
    return log2(x);
  }
};

struct Log10 {
  template <typename T>
  __device__ T operator()(T x) {
    return log10(x);
  }
};

struct Log1p {
  template <typename T>
  __device__ T operator()(T x) {
    return log1p(x);
  }
};

struct Round {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      return {rint(cuCrealf(x)), rint(cuCimagf(x))};
    } else {
      return rint(x);
    }
  }
};

} // namespace mlx::core::mxcuda
