// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/utils.h"

#include <fmt/format.h>

namespace mlx::core {

void check_cuda_error(cudaError_t err) {
  if (err != cudaSuccess) {
    throw std::runtime_error(
        fmt::format("cudaSetDevice failed: %s", cudaGetErrorString(err)));
  }
}

} // namespace mlx::core
