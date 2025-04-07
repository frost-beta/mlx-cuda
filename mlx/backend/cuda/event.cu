// Copyright © 2024 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/event.h"
#include "mlx/backend/cuda/utils.h"
#include "mlx/event.h"
#include "mlx/scheduler.h"

#include <nvtx3/nvtx3.hpp>
#include <cuda/atomic>

namespace mlx::core {

namespace mxcuda {

namespace {

__host__ __device__ void event_wait(SharedEvent::Atomic* ac, uint64_t value) {
  uint64_t current;
  while ((current = ac->load()) < value) {
    ac->wait(current);
  }
}

__host__ __device__ void event_signal(SharedEvent::Atomic* ac, uint64_t value) {
  ac->store(value);
  ac->notify_all();
}

__global__ void event_wait_kernel(SharedEvent::Atomic* ac, uint64_t value) {
  event_wait(ac, value);
}

__global__ void event_signal_kernel(SharedEvent::Atomic* ac, uint64_t value) {
  event_signal(ac, value);
}

} // namespace

SharedEvent::SharedEvent() {
  // Allocate cuda::atomic on managed memory.
  Atomic* ac;
  CHECK_CUDA_ERROR(cudaMallocManaged(&ac, sizeof(Atomic)));
  new (ac) Atomic(0);
  ac_ = std::shared_ptr<Atomic>(ac, [](Atomic* ptr) {
    ptr->~atomic<uint64_t>();
    cudaFree(ptr);
  });
}

void SharedEvent::wait(uint64_t value) {
  nvtx3::scoped_range r("mxcuda::SharedEvent::wait");
  event_wait(ac_.get(), value);
}

void SharedEvent::wait(Stream stream, uint64_t value) {
  nvtx3::scoped_range r("mxcuda::SharedEvent::wait(stream)");
  if (stream.device == mlx::core::Device::cpu) {
    scheduler::enqueue(
        stream, [ac = ac_, value]() { event_wait(ac.get(), value); });
  } else {
    mxcuda::get_command_encoder(stream).launch_kernel_sequencially(
        [this, value](cudaStream_t s) {
          event_wait_kernel<<<1, 1, 0, s>>>(ac_.get(), value);
        });
  }
}

void SharedEvent::signal(Stream stream, uint64_t value) {
  nvtx3::scoped_range r("mxcuda::SharedEvent::signal");
  if (stream.device == mlx::core::Device::cpu) {
    scheduler::enqueue(
        stream, [ac = ac_, value]() { event_signal(ac.get(), value); });
  } else {
    mxcuda::get_command_encoder(stream).launch_kernel_sequencially(
        [this, value](cudaStream_t s) {
          event_signal_kernel<<<1, 1, 0, s>>>(ac_.get(), value);
        });
  }
}

bool SharedEvent::is_signaled(uint64_t value) const {
  nvtx3::scoped_range r("mxcuda::SharedEvent::is_signaled");
  return ac_->load() >= value;
}

} // namespace mxcuda

Event::Event(Stream stream) : stream_(stream) {
  event_ = std::shared_ptr<void>(new mxcuda::SharedEvent(), [](void* ptr) {
    delete static_cast<mxcuda::SharedEvent*>(ptr);
  });
}

void Event::wait() {
  static_cast<mxcuda::SharedEvent*>(event_.get())->wait(value());
}

void Event::wait(Stream stream) {
  static_cast<mxcuda::SharedEvent*>(event_.get())->wait(stream, value());
}

void Event::signal(Stream stream) {
  static_cast<mxcuda::SharedEvent*>(event_.get())->signal(stream, value());
}

bool Event::is_signaled() const {
  return static_cast<mxcuda::SharedEvent*>(event_.get())->is_signaled(value());
}

} // namespace mlx::core
