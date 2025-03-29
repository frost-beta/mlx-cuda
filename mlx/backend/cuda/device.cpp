// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/utils.h"
#include "mlx/backend/metal/metal.h"

#include <fmt/format.h>
#include <unordered_map>

namespace mlx::core {

namespace mxcuda {

DeviceStream::DeviceStream(Stream stream) {
  int device = stream.device.index;
  set_cuda_device(device);
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
}

DeviceStream::~DeviceStream() {
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream_));
}

cudaStream_t DeviceStream::schedule_cuda_stream() {
  // TODO: Return a stream that maximizes parallelism.
  return stream_;
}

cudaStream_t DeviceStream::last_cuda_stream() {
  return stream_;
}

void DeviceStream::add_host_callback(std::function<void()> func) {
  CHECK_CUDA_ERROR(cudaLaunchHostFunc(
      last_cuda_stream(),
      [](void* ptr) {
        auto* func = static_cast<std::function<void()>*>(ptr);
        (*func)();
        delete func;
      },
      new std::function<void()>(std::move(func))));
}

void CommandEncoder::prefetch_memory(const array& arr) {
  // TODO: Profile whether prefetching the whole buffer would be faster.
  const void* data = arr.data<void>();
  size_t size = arr.data_size() * arr.itemsize();
  if (data && size > 0) {
    // TODO: Use a stream that maximizes parallelism.
    CHECK_CUDA_ERROR(
        cudaMemPrefetchAsync(data, size, device_, stream_.last_cuda_stream()));
  }
}

Device::Device(int device) {
  // Validate the requirements of device.
  int attr = 0;
  cudaDeviceGetAttribute(&attr, cudaDevAttrConcurrentManagedAccess, device);
  if (attr != 1) {
    throw std::runtime_error(fmt::format(
        "Device {} does not support synchronization in managed memory.",
        device));
  }
  // The cublasLt handle is used for matmul.
  set_cuda_device(device);
  cublasLtCreate(&lt_);
}

Device::~Device() {
  cublasLtDestroy(lt_);
}

Device& device(mlx::core::Device device) {
  static std::vector<Device> devices;
  for (int i = devices.size(); i <= device.index; ++i) {
    devices.push_back(Device(i));
  }
  return devices[device.index];
}

DeviceStream& get_stream(Stream stream) {
  return get_command_encoder(stream).stream();
}

CommandEncoder& get_command_encoder(Stream stream) {
  static std::unordered_map<int, CommandEncoder> encoder_map;
  auto it = encoder_map.find(stream.index);
  if (it == encoder_map.end()) {
    it = encoder_map.emplace(stream.index, stream).first;
  }
  return it->second;
}

} // namespace mxcuda

namespace metal {

void new_stream(Stream stream) {
  // Ensure the static stream objects get created.
  mxcuda::get_command_encoder(stream);
}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info() {
  throw std::runtime_error(
      "[metal::device_info] Not implemented in CUDA backend.");
};

} // namespace metal

} // namespace mlx::core
