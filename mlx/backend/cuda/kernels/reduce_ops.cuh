// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/kernels/utils.cuh"

namespace mlx::core::mxcuda {

struct And {
  __device__ bool operator()(bool a, bool b) {
    return a && b;
  }
};

struct Or {
  __device__ bool operator()(bool a, bool b) {
    return a || b;
  }
};

struct Sum {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return a + b;
  }
};

struct Prod {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return a * b;
  }
};

struct Min {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return a < b ? a : b;
  }
};

struct Max {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return a > b ? a : b;
  }
};

template <typename Op, typename T>
struct ReduceInit;

template <typename T>
struct ReduceInit<And, T> {
  using type = bool;
  static constexpr bool value = true;
};

template <typename T>
struct ReduceInit<Or, T> {
  using type = bool;
  static constexpr bool value = false;
};

template <typename T>
struct ReduceInit<Sum, T> {
  using type = T;
  static constexpr auto value = zero_value<T>();
};

template <typename T>
struct ReduceInit<Prod, T> {
  using type = T;
  static constexpr auto value = one_value<T>();
};

template <typename T>
struct ReduceInit<Min, T> {
  using type = T;
  static constexpr auto value = Limits<T>::max;
};

template <typename T>
struct ReduceInit<Max, T> {
  using type = T;
  static constexpr auto value = Limits<T>::min;
};

} // namespace mlx::core::mxcuda
