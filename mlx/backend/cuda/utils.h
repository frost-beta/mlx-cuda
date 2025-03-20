// Copyright © 2025 Apple Inc.

#pragma once

#include <cuda_runtime.h>

namespace mlx::core {

void check_cuda_error(const char* name, cudaError_t err);

// Throw exception if the cuda API does not succeed.
#define CHECK_CUDA_ERROR(cmd) check_cuda_error(#cmd, (cmd))

} // namespace mlx::core
