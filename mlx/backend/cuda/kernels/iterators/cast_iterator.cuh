// Copyright © 2025 Apple Inc.

#pragma once

#include <thrust/iterator/transform_iterator.h>

namespace mlx::core::cu {

template <typename SrcT, typename DstT>
struct CastOp {
  __device__ DstT operator()(SrcT x) {
    return static_cast<DstT>(x);
  }
};

// Return an iterator that static_cast the value to T.
template <typename T, typename U>
__host__ __device__ auto make_cast_iterator(U* it) {
  return thrust::make_transform_iterator(it, CastOp<U, T>{});
}

template <typename T, typename Iterator>
__host__ __device__ auto make_cast_iterator(Iterator it) {
  return thrust::make_transform_iterator(
      it, CastOp<typename Iterator::value_type, T>{});
}

} // namespace mlx::core::cu
