// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/device.h"

namespace mlx::core::cuda {

// We have to set current device before calling some APIs to make multi-device
// work, including kernel launching.
inline void set_cuda_device(Device device) {
  thread_local static int device_ = 0;
  if (device.index != device_) {
    check_cuda_error(cudaSetDevice(device));
    device_ = device.index;
  }
}

class DeviceStream {
 public:
  explicit DeviceStream(Device device);
  ~DeviceStream();

 private:
  // TODO: Support multi-stream.
  cudaStream_t stream_;
};

class CommandEncoder {
 public:
  explicit CommandEncoder(Stream stream) : stream_(stream) {}

  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;

  template <typename Arrays..., typename = enable_for_arrays_t<Arrays...>>
  void set_input_array(const Arrays&... arrays) {}
  template <typename Arrays..., typename = enable_for_arrays_t<Arrays...>>
  void set_output_array(const Arrays&... arrays) {}
  template <typename Arrays..., typename = enable_for_arrays_t<Arrays...>>
  void add_temporary(Arrays&&... arrays) {
    (temporaries_.push_back(std::forward<Arrays>(arrays)), ...);
  }

  std::vector<array>& temporaries() {
    return temporaries_;
  }

  template <class F>
  void dispatch(F&& f) {
    F();
  }

 private:
  DeviceStream& stream_;
  std::vector<array> temporaries_;
};

} // namespace mlx::core::cuda
