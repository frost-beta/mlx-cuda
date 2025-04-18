// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/allocator.h"
#include "mlx/backend/cuda/utils.h"
#include "mlx/backend/metal/metal.h"

#include <fmt/format.h>
#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace mxcuda {

DeviceStream::DeviceStream(Device& device, Stream s) : device_(device) {
  device_.make_current();
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
}

DeviceStream::~DeviceStream() {
  // The cuda stream is leaked on exit: it is unknown when cuda runtime shuts
  // down, and it could happen before we clean up streams and would crash.
  // CHECK_CUDA_ERROR(cudaStreamDestroy(stream_));
}

void DeviceStream::synchronize() {
  // TODO: Wait for all cuda streams in mlx stream.
  cudaStreamSynchronize(stream_);
}

cudaStream_t DeviceStream::schedule_cuda_stream() {
  // TODO: Return a stream that maximizes parallelism.
  return stream_;
}

cudaStream_t DeviceStream::last_cuda_stream() {
  return stream_;
}

void DeviceStream::finalize() {
  if (!retained_.empty()) {
    add_host_callback([retained = std::move(retained_)]() {
      nvtx3::mark("mxcuda::DeviceStream::finalize");
    });
  }
}

CommandEncoder& DeviceStream::get_encoder() {
  if (!encoder_) {
    encoder_ = std::make_unique<CommandEncoder>(*this);
  }
  return *encoder_;
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

Device::Device(int device) : device_(device) {
  // Validate the requirements of device.
  int attr = 0;
  cudaDeviceGetAttribute(&attr, cudaDevAttrConcurrentManagedAccess, device_);
  if (attr != 1) {
    throw std::runtime_error(fmt::format(
        "Device {} does not support synchronization in managed memory.",
        device_));
  }
  // The cublasLt handle is used for matmul.
  make_current();
  cublasLtCreate(&lt_);
}

Device::~Device() {
  cublasLtDestroy(lt_);
}

void Device::make_current() {
  // We need to set/get current CUDA device very frequently, cache it to reduce
  // actual calls of CUDA APIs. This function assumes single-thread in host.
  static int current = 0;
  if (current != device_) {
    CHECK_CUDA_ERROR(cudaSetDevice(device_));
    current = device_;
  }
}

DeviceStream& Device::get_stream(Stream s) {
  auto it = streams_.find(s.index);
  if (it == streams_.end()) {
    it = streams_.try_emplace(s.index, *this, s).first;
  }
  return it->second;
}

CommandEncoder::CommandEncoder(DeviceStream& s)
    : device_(s.device()), stream_(s) {}

void CommandEncoder::end_eval() {
  if (!temporaries_.empty()) {
    stream_.retain_until_completion(std::move(temporaries_));
  }
}

Device& device(mlx::core::Device device) {
  static std::unordered_map<int, Device> devices;
  auto it = devices.find(device.index);
  if (it == devices.end()) {
    it = devices.try_emplace(device.index, device.index).first;
  }
  return it->second;
}

DeviceStream& get_stream(Stream s) {
  return device(s.device).get_stream(s);
}

CommandEncoder& get_command_encoder(Stream s) {
  return get_stream(s).get_encoder();
}

} // namespace mxcuda

namespace metal {

void new_stream(Stream s) {
  // Ensure the static stream objects get created.
  mxcuda::get_command_encoder(s);
}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info() {
  throw std::runtime_error(
      "[metal::device_info] Not implemented in CUDA backend.");
};

} // namespace metal

} // namespace mlx::core
