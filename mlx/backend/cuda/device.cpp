// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/utils.h"

#include <cuda/atomic>
#include <unordered_map>

namespace mlx::core::cuda {

DeviceStream::DeviceStream(Stream stream) : device_(stream.device) {
  set_cuda_device(device_);
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
  // Validate the requirements of device.
  // TODO: Validate per-device instead of per-stream.
  if (!::cuda::atomic<uint64_t>::is_always_lock_free) {
    throw std::runtime_error(
        "Synchronization not supported for managed memory: "
        "cuda::atomic<uint64_t>::is_always_lock_free is false.");
  }
  int a = 0;
  cudaDeviceGetAttribute(&a, cudaDevAttrConcurrentManagedAccess, device_.index);
  if (a != 1) {
    throw std::runtime_error(
        "Synchronization not supported for managed memory: "
        "cudaDevAttrConcurrentManagedAccess is 0.");
  }
}

DeviceStream::~DeviceStream() {
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream_));
}

cudaStream_t DeviceStream::schedule_cuda_stream() {
  set_cuda_device(device_);
  // TODO: Return a stream that maximizes parallelism.
  return stream_;
}

CommandEncoder& get_command_encoder(Stream stream) {
  static std::unordered_map<int, CommandEncoder> encoder_map;
  auto it = encoder_map.find(stream.index);
  if (it == encoder_map.end()) {
    it = encoder_map.emplace(stream.index, stream).first;
  }
  return it->second;
}

} // namespace mlx::core::cuda
