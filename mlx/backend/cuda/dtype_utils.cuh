// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/dtype_utils.h"

#include <cuComplex.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace mlx::core {

// Maps Dtypes to C++ types.
template <typename T>
struct CTypeToCudaType {
  using type = T;
};

template <>
struct CTypeToCudaType<float16_t> {
  using type = __half;
};

template <>
struct CTypeToCudaType<bfloat16_t> {
  using type = __nv_bfloat16;
};

template <>
struct CTypeToCudaType<complex64_t> {
  using type = cuComplex;
};

// Replace native type to CUDA types.
#define MLX_SWITCH_CUDA_TYPES(TYPE, NAME, CTYPE_ALIAS, ...)               \
  MLX_SWITCH_ALL_TYPES(TYPE, NAME, CTYPE_NATIVE, [&]() {                  \
    using CTYPE_ALIAS = ::mlx::core::CTypeToCudaType<CTYPE_NATIVE>::type; \
    return __VA_ARGS__();                                                 \
  })

} // namespace mlx::core
