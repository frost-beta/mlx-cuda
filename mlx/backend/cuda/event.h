// Copyright © 2025 Apple Inc.

#pragma once

#include <cuda/atomic>

namespace mlx::core::mxcuda {

// Event that can synchronize between CPU and GPU.
class SharedEvent {
 public:
  // The cuda::atomic can synchronize between CPU and GPU, with a little
  // performance penalty: some ops can take more than 1ms.
  using Atomic = cuda::atomic<uint64_t>;

  SharedEvent();

  SharedEvent(const SharedEvent&) = delete;
  SharedEvent& operator=(const SharedEvent&) = delete;

  void wait(uint64_t value);
  void wait(Stream stream, uint64_t value);
  void signal(Stream stream, uint64_t value);
  bool is_signaled(uint64_t value) const;

 private:
  std::shared_ptr<Atomic> ac_;
};

} // namespace mlx::core::mxcuda
