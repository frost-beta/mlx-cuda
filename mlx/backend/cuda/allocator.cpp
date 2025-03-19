// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/allocator.h"

#include <cuda_runtime.h>
#include <fmt/format.h>

namespace mlx::core {

namespace allocator {

Allocator& allocator() {
  return cuda::allocator();
}

void* Buffer::raw_ptr() {
  if (!ptr_) {
    return nullptr;
  }
  return static_cast<cuda::CudaBuffer*>(ptr_)->data;
}

} // namespace allocator

namespace cuda {

Buffer CudaAllocator::malloc(size_t size, bool) {
  auto* buf = new CudaBuffer{nullptr, size};
  cudaError_t err = cudaMallocManaged(&buf->data, size);
  if (err != cudaSuccess && err != cudaErrorMemoryAllocation) {
    throw std::runtime_error(
        fmt::format("cudaMallocManaged failed: %s", cudaGetErrorString(err)));
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
  return static_cast<cuda::CudaBuffer*>(buffer.ptr())->size;
}

CudaAllocator& allocator() {
  // By creating the |allocator_| on heap, the destructor of CudaAllocator
  // will not be called on exit and buffers in the cache will be leaked. This
  // can save some time at program exit.
  static CudaAllocator* allocator_ = new CudaAllocator;
  return *allocator_;
}

} // namespace cuda

} // namespace mlx::core
