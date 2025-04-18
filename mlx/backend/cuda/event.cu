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

void SharedEvent::wait(Stream s, uint64_t value) {
  nvtx3::scoped_range r("mxcuda::SharedEvent::wait(s)");
  if (s.device == mlx::core::Device::cpu) {
    scheduler::enqueue(s, [*this, value]() mutable { wait(value); });
  } else {
    auto& encoder = mxcuda::get_command_encoder(s);
    encoder.launch_kernel(
        encoder.stream().last_cuda_stream(),
        [this, value](cudaStream_t stream) {
          event_wait_kernel<<<1, 1, 0, stream>>>(ac_.get(), value);
        });
  }
}

void SharedEvent::signal(uint64_t value) {
  nvtx3::scoped_range r("mxcuda::SharedEvent::signal");
  event_signal(ac_.get(), value);
}

void SharedEvent::signal(Stream s, uint64_t value) {
  nvtx3::scoped_range r("mxcuda::SharedEvent::signal(s)");
  if (s.device == mlx::core::Device::cpu) {
    scheduler::enqueue(s, [*this, value]() mutable { signal(value); });
  } else {
    auto& encoder = mxcuda::get_command_encoder(s);
    encoder.launch_kernel(
        encoder.stream().last_cuda_stream(),
        [this, value](cudaStream_t stream) {
          event_signal_kernel<<<1, 1, 0, stream>>>(ac_.get(), value);
        });
  }
}

bool SharedEvent::is_signaled(uint64_t value) const {
  nvtx3::scoped_range r("mxcuda::SharedEvent::is_signaled");
  return ac_->load() >= value;
}

CudaEvent::CudaEvent() {
  CHECK_CUDA_ERROR(cudaEventCreateWithFlags(
      &event_, cudaEventDisableTiming | cudaEventBlockingSync));
}

CudaEvent::~CudaEvent() {
  CHECK_CUDA_ERROR(cudaEventDestroy(event_));
}

void CudaEvent::wait() {
  nvtx3::scoped_range r("mxcuda::CudaEvent::wait");
  cudaEventSynchronize(event_);
}

void CudaEvent::wait(Stream s) {
  nvtx3::scoped_range r("mxcuda::CudaEvent::wait(s)");
  assert(s.device != mlx::core::Device::cpu);
  cudaStreamWaitEvent(get_stream(s).last_cuda_stream(), event_);
}

void CudaEvent::record(Stream s) {
  nvtx3::scoped_range r("mxcuda::CudaEvent::record(s)");
  assert(s.device != mlx::core::Device::cpu);
  cudaEventRecord(event_, get_stream(s).last_cuda_stream());
}

bool CudaEvent::completed() const {
  return cudaEventQuery(event_) == cudaSuccess;
}

} // namespace mxcuda

namespace {

struct EventImpl {
  std::unique_ptr<mxcuda::SharedEvent> shared;
  std::unique_ptr<mxcuda::CudaEvent> cuda;

  bool is_created() const {
    return cuda || shared;
  }

  void ensure_created(Stream s, uint64_t signal_value) {
    if (is_created()) {
      return;
    }
    if (s.device == mlx::core::Device::cpu || signal_value > 1) {
      nvtx3::mark("Using slow SharedEvent");
      shared = std::make_unique<mxcuda::SharedEvent>();
    } else {
      cuda = std::make_unique<mxcuda::CudaEvent>();
    }
  }
};

} // namespace

Event::Event(Stream s) : stream_(s) {
  event_ = std::shared_ptr<void>(
      new EventImpl(), [](void* ptr) { delete static_cast<EventImpl*>(ptr); });
}

void Event::wait() {
  auto* event = static_cast<EventImpl*>(event_.get());
  if (!event->is_created()) {
    // The code calls wait() before signal(), with cuda event it would be
    // treated as if already signaled.
    event->shared = std::make_unique<mxcuda::SharedEvent>();
  } else {
    event->ensure_created(stream_, value());
  }
  if (event->shared) {
    event->shared->wait(value());
  } else {
    assert(value() <= 1);
    event->cuda->wait();
  }
}

void Event::wait(Stream s) {
  auto* event = static_cast<EventImpl*>(event_.get());
  if (!event->is_created()) {
    // The code calls wait() before signal(), with cuda event it would be
    // treated as if already signaled.
    event->shared = std::make_unique<mxcuda::SharedEvent>();
  } else {
    event->ensure_created(s, value());
  }
  if (event->shared) {
    event->shared->wait(s, value());
  } else {
    assert(value() <= 1);
    event->cuda->wait(s);
  }
}

void Event::signal(Stream s) {
  auto* event = static_cast<EventImpl*>(event_.get());
  event->ensure_created(s, value());
  if (event->shared) {
    event->shared->signal(s, value());
  } else {
    event->cuda->record(s);
  }
}

bool Event::is_signaled() const {
  auto* event = static_cast<EventImpl*>(event_.get());
  if (!event->is_created()) {
    return true;
  }
  if (event->shared) {
    return event->shared->is_signaled(value());
  } else {
    assert(value() <= 1);
    return event->cuda->completed();
  }
}

} // namespace mlx::core
