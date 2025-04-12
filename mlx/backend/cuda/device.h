// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/cuda/utils.h"
#include "mlx/stream.h"

#include <cublasLt.h>
#include <thrust/execution_policy.h>

#include <unordered_map>

namespace mlx::core::mxcuda {

class Device;
class CommandEncoder;

// A stream in MLX consists of multiple CUDA stream.
class DeviceStream {
 public:
  DeviceStream(Device& device, Stream s);
  ~DeviceStream();

  DeviceStream(const DeviceStream&) = delete;
  DeviceStream& operator=(const DeviceStream&) = delete;

  // Wait until all current tasks finish.
  void synchronize();

  // Return a CUDA stream for launching kernels.
  cudaStream_t schedule_cuda_stream();

  // Return the last stream used.
  cudaStream_t last_cuda_stream();

  // Keep the buffers alive until at least current tasks are finished.
  template <typename Container>
  void retain_until_completion(Container buffers) {
    std::move(buffers.begin(), buffers.end(), std::back_inserter(retained_));
    // TODO: We don't want to retain the buffers for too long, which increases
    // memory usage, and we don't want to release them too soon, which delays
    // the kernel execution. Find a strategy that balances performance.
    if (retained_.size() > 32) {
      finalize();
    }
  }

  // Clear the retained arrays when current tasks are finished.
  void finalize();

  CommandEncoder& get_encoder();

  Device& device() {
    return device_;
  }

 private:
  // Run the function in host after last launched work finishes. This call adds
  // at least 20µs latency in cuda stream, so only use it when necessary.
  void add_host_callback(std::function<void()> func);

  Device& device_;
  cudaStream_t stream_;
  std::unique_ptr<CommandEncoder> encoder_;
  std::vector<std::shared_ptr<array::Data>> retained_;
};

class Device {
 public:
  explicit Device(int device);
  ~Device();

  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  // Make this instance the current CUDA device, required by some CUDA calls.
  void make_current();

  DeviceStream& get_stream(Stream s);

  int cuda_device() const {
    return device_;
  }

  cublasLtHandle_t lt_handle() const {
    return lt_;
  }

 private:
  int device_;
  cublasLtHandle_t lt_;
  std::unordered_map<int, DeviceStream> streams_;
};

class CommandEncoder {
 public:
  explicit CommandEncoder(DeviceStream& stream);

  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;

  template <typename... Arrays, typename = enable_for_arrays_t<Arrays...>>
  void set_input_array(const Arrays&... arrays) {}

  template <typename... Arrays, typename = enable_for_arrays_t<Arrays...>>
  void set_output_array(const Arrays&... arrays) {}

  void add_temporary(const array& arr) {
    temporaries_.push_back(arr.data_shared_ptr());
  }

  // Current temporaries will no longer be used.
  void end_eval();

  // Setup everything for launching cuda kernel in |fun|.
  template <typename F>
  void launch_kernel(F&& fun) {
    launch_kernel(stream_.schedule_cuda_stream(), std::forward<F>(fun));
  }

  template <typename F>
  void launch_kernel(cudaStream_t stream, F&& fun) {
    device_.make_current();
    fun(stream);
    check_cuda_error("kernel launch", cudaGetLastError());
  }

  Device& device() {
    return device_;
  }

  DeviceStream& stream() {
    return stream_;
  }

 private:
  Device& device_;
  DeviceStream& stream_;
  std::vector<std::shared_ptr<array::Data>> temporaries_;
};

Device& device(mlx::core::Device device);
DeviceStream& get_stream(Stream s);
CommandEncoder& get_command_encoder(Stream s);

// Return an execution policy that does not sync for result.
// Note that not all thrust APIs support async policy, confirm before using.
inline auto thrust_policy(cudaStream_t stream) {
  // TODO: Connect thrust's custom allocator with mlx's allocator.
  return thrust::cuda::par_nosync.on(stream);
}

} // namespace mlx::core::mxcuda
