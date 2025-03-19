// Copyright © 2024 Apple Inc.

#include "mlx/backend/cuda/utils.h"
#include "mlx/event.h"
#include "mlx/scheduler.h"

#include <cuda/atomic>

namespace mlx::core {

Event::Event(Stream stream) : stream_(stream) {
  // TODO: Move the check to initialization.
  if (!cuda::atomic<uint64_t>::is_always_lock_free) {
    throw std::runtime_error(
        "Synchronization not supported for managed memory.");
  }
  // Allocate cuda::atomic on managed memory.
  cuda::atomic<uint64_t>* ac;
  check_cuda_error(cudaMallocManaged(&ac, sizeof(cuda::atomic<uint64_t>)));
  new(ac) std::atomic<uint64_t>(0);
  // Store it in a shared_ptr.
  auto dtor = [](void* ptr) {
    static_cast<cuda::atomic<uint64_t>*>(ptr)->~atomic<uint64_t>();
    cudaFree(ptr);
  };
  event_ = std::shared_ptr<void>(ac, dtor);
}

void Event::wait() {
  auto* ac = static_cast<cuda::atomic<uint64_t>*>(event_.get());
  uint64_t current;
  while ((current = ac->load()) < value()) {
    ac->wait(current);
  }
}

void Event::signal() {
  auto* ac = static_cast<cuda::atomic<uint64_t>*>(event_.get());
  ac->store(value());
  ac->notify_all();
}

void Event::wait(Stream stream) {
  scheduler::enqueue(stream, [*this]() mutable { wait(); });
}

void Event::signal(Stream stream) {
  scheduler::enqueue(stream, [*this]() mutable { signal(); });
}

bool Event::is_signaled() const {
  auto* ac = static_cast<cuda::atomic<uint64_t>*>(event_.get());
  return ac->load() >= value();
}
} // namespace mlx::core
