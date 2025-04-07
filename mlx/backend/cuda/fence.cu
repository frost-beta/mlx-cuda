// Copyright © 2024 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/utils.h"
#include "mlx/fence.h"
#include "mlx/scheduler.h"

#include <nvtx3/nvtx3.hpp>
#include <cuda/atomic>

namespace mlx::core {

namespace {

// The cuda::atomic can synchronize between CPU and GPU, with a little
// performance penalty: some ops can take more than 1ms.
using CudaAtomic = cuda::atomic<uint32_t>;

__host__ __device__ void atomic_wait(CudaAtomic* ac, uint32_t value) {
  uint32_t current;
  while ((current = ac->load()) < value) {
    ac->wait(current);
  }
}

__host__ __device__ void atomic_signal(CudaAtomic* ac, uint32_t value) {
  ac->store(value);
  ac->notify_all();
}

__global__ void atomic_wait_kernel(CudaAtomic* ac, uint32_t value) {
  atomic_wait(ac, value);
}

__global__ void atomic_signal_kernel(CudaAtomic* ac, uint32_t value) {
  atomic_signal(ac, value);
}

} // namespace

struct FenceImpl {
  std::shared_ptr<CudaAtomic> ac;
  uint32_t count;
};

Fence::Fence(Stream stream) {
  // Allocate cuda::atomic on managed memory.
  CudaAtomic* ptr;
  CHECK_CUDA_ERROR(cudaMallocManaged(&ptr, sizeof(CudaAtomic)));
  new (ptr) CudaAtomic(0);
  // Store it in a shared_ptr.
  std::shared_ptr<CudaAtomic> ac(ptr, [](CudaAtomic* ptr) {
    ptr->~CudaAtomic();
    cudaFree(ptr);
  });
  // Allocate FenceImpl on host memory.
  fence_ = std::shared_ptr<void>(
      new FenceImpl{std::move(ac), 0},
      [](void* ptr) { delete static_cast<FenceImpl*>(ptr); });
}

void Fence::wait(Stream stream, const array&) {
  nvtx3::scoped_range r("Fence::wait");
  auto* fence = static_cast<FenceImpl*>(fence_.get());
  if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [ac = fence->ac, value = fence->count]() {
      atomic_wait(ac.get(), value);
    });
  } else {
    mxcuda::get_command_encoder(stream).launch_kernel_sequencially(
        [ac = fence->ac, value = fence->count](cudaStream_t s) {
          atomic_wait_kernel<<<1, 1, 0, s>>>(ac.get(), value);
        });
  }
}

void Fence::update(Stream stream, const array&) {
  nvtx3::scoped_range r("Fence::update");
  auto* fence = static_cast<FenceImpl*>(fence_.get());
  fence->count++;
  if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [ac = fence->ac, value = fence->count]() {
      atomic_signal(ac.get(), value);
    });
  } else {
    mxcuda::get_command_encoder(stream).launch_kernel_sequencially(
        [ac = fence->ac, value = fence->count](cudaStream_t s) {
          atomic_signal_kernel<<<1, 1, 0, s>>>(ac.get(), value);
        });
  }
}

} // namespace mlx::core
