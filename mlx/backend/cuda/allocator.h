// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/allocator.h"

namespace mlx::core::cuda {

using allocator::Buffer;

// Stores cuda-managed memory.
struct CudaBuffer {
  void* data;
  size_t size;
};

class CudaAllocator : public allocator::Allocator {
 public:
  Buffer malloc(size_t size, bool allow_swap) override;
  void free(Buffer buffer) override;
  size_t size(Buffer buffer) const override;

 private:
  CudaAllocator() = default;
  friend CudaAllocator& allocator();
};

CudaAllocator& allocator();

} // namespace mlx::core::cuda
