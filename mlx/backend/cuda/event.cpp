// Copyright © 2024 Apple Inc.

#include "mlx/event.h"
#include "mlx/backend/cuda/utils.h"
#include "mlx/scheduler.h"

#include <cuda/atomic>

namespace mlx::core {

Event::Event(Stream stream) : stream_(stream) {
  // Allocate cuda::atomic on managed memory.
  cuda::atomic<uint64_t>* ac;
  CHECK_CUDA_ERROR(cudaMallocManaged(&ac, sizeof(cuda::atomic<uint64_t>)));
  new (ac) std::atomic<uint64_t>(0);
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
  if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [*this]() mutable { wait(); });
  } else {
    throw std::runtime_error("Event::wait not implemented for GPU yet.");
  }
}

void Event::signal(Stream stream) {
  if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [*this]() mutable { signal(); });
  } else {
    throw std::runtime_error("Event::signal not implemented for GPU yet.");
  }
}

bool Event::is_signaled() const {
  auto* ac = static_cast<cuda::atomic<uint64_t>*>(event_.get());
  return ac->load() >= value();
}
} // namespace mlx::core
