// Copyright © 2025 Apple Inc.

#pragma once

#include <cuda_runtime.h>

namespace mlx::core {

void check_cuda_error(cudaError_t err);

} // namespace mlx::core
