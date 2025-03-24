// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"

#include <cuda_runtime.h>
#include <type_traits>

namespace mlx::core {

void check_cuda_error(const char* name, cudaError_t err);

// Throw exception if the cuda API does not succeed.
#define CHECK_CUDA_ERROR(cmd) check_cuda_error(#cmd, (cmd))

// Return the 3d block_dim fit for total_threads.
dim3 get_block_dim(dim3 total_threads, int pow2 = 10);

// Return the 2d block_dim needed for 1d block of num_threads.
dim3 get_2d_num_blocks(
    const Shape& shape,
    const Strides& strides,
    size_t num_threads);

std::string get_primitive_string(Primitive* primitive);

} // namespace mlx::core
