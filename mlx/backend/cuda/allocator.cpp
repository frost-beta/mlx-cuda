// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/allocator.h"

#include <cuda_runtime.h>
#include <fmt/format.h>

namespace mlx::core {

namespace mxcuda {

Buffer CudaAllocator::malloc(size_t size, bool) {
  auto* buf = new CudaBuffer{nullptr, size};
  cudaError_t err = cudaMallocManaged(&buf->data, size);
  if (err != cudaSuccess && err != cudaErrorMemoryAllocation) {
    throw std::runtime_error(
        fmt::format("cudaMallocManaged failed: {}", cudaGetErrorString(err)));
  }
  return Buffer{buf};
}

void CudaAllocator::free(Buffer buffer) {
  auto* buf = static_cast<CudaBuffer*>(buffer.ptr());
  if (!buf) {
    return;
  }
  cudaFree(buf->data);
  delete buf;
}

size_t CudaAllocator::size(Buffer buffer) const {
  auto* buf = static_cast<CudaBuffer*>(buffer.ptr());
  if (!buf) {
    return 0;
  }
  return static_cast<CudaBuffer*>(buffer.ptr())->size;
}

CudaAllocator& allocator() {
  // By creating the |allocator_| on heap, the destructor of CudaAllocator
  // will not be called on exit and buffers in the cache will be leaked. This
  // can save some time at program exit.
  static CudaAllocator* allocator_ = new CudaAllocator;
  return *allocator_;
}

} // namespace mxcuda

namespace allocator {

Allocator& allocator() {
  return mxcuda::allocator();
}

void* Buffer::raw_ptr() {
  if (!ptr_) {
    return nullptr;
  }
  return static_cast<mxcuda::CudaBuffer*>(ptr_)->data;
}

} // namespace allocator

namespace metal {

void clear_cache() {}

size_t get_active_memory() {
  return 0;
}
size_t get_peak_memory() {
  return 0;
}
void reset_peak_memory() {}
size_t get_cache_memory() {
  return 0;
}
size_t set_memory_limit(size_t, bool) {
  return 0;
}
size_t set_cache_limit(size_t) {
  return 0;
}
size_t set_wired_limit(size_t) {
  return 0;
}

} // namespace metal

} // namespace mlx::core
