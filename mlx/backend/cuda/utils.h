// Copyright © 2025 Apple Inc.

#pragma once

#include <cuda_runtime.h>

namespace mlx::core {

#define CHECK_CUDA_ERROR(cmd) check_cuda_error(#cmd, (cmd))

void check_cuda_error(const char* name, cudaError_t err);

} // namespace mlx::core
