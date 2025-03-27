// Copyright © 2025 Apple Inc.

#pragma once

#include <thrust/iterator/iterator_adaptor.h>

namespace mlx::core::mxcuda {

// Always return the same iterator after advancements.
template <typename Iterator>
class repeat_iterator
    : public thrust::iterator_adaptor<repeat_iterator<Iterator>, Iterator> {
 public:
  using super_t = thrust::iterator_adaptor<repeat_iterator<Iterator>, Iterator>;
  using reference = typename super_t::reference;
  using difference_type = typename super_t::difference_type;

  using super_t::super_t;

 private:
  friend class thrust::iterator_core_access;

  __host__ __device__ void advance(difference_type n) {}
  __host__ __device__ void increment() {}
  __host__ __device__ void decrement() {}
};

template <typename Iterator>
__host__ __device__ auto make_repeat_iterator(Iterator it) {
  return repeat_iterator<Iterator>(it);
}

} // namespace mlx::core::mxcuda
