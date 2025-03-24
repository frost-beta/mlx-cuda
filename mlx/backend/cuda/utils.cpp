// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/utils.h"
#include "mlx/primitives.h"

#include <fmt/format.h>

namespace mlx::core {

void check_cuda_error(const char* name, cudaError_t err) {
  if (err != cudaSuccess) {
    throw std::runtime_error(
        fmt::format("{} failed: {}", name, cudaGetErrorString(err)));
  }
}

// TODO: The implementation is identical to meta/utils.cpp .
dim3 get_block_dim(dim3 total_threads, int pow2) {
  int pows[3] = {0, 0, 0};
  int sum = 0;
  while (true) {
    int presum = sum;
    // Check all the pows
    if (total_threads.x >= (1 << (pows[0] + 1))) {
      pows[0]++;
      sum++;
    }
    if (sum == 10) {
      break;
    }
    if (total_threads.y >= (1 << (pows[1] + 1))) {
      pows[1]++;
      sum++;
    }
    if (sum == 10) {
      break;
    }
    if (total_threads.z >= (1 << (pows[2] + 1))) {
      pows[2]++;
      sum++;
    }
    if (sum == presum || sum == pow2) {
      break;
    }
  }
  return {1u << pows[0], 1u << pows[1], 1u << pows[2]};
}

dim3 get_2d_num_blocks(
    const Shape& shape,
    const Strides& strides,
    size_t num_threads) {
  size_t grid_x = 1;
  size_t grid_y = 1;
  for (int i = 0; i < shape.size(); ++i) {
    if (strides[i] == 0) {
      continue;
    }

    // No need to add this shape we can just remove it from the num_threads.
    if (num_threads % shape[i] == 0) {
      num_threads /= shape[i];
      continue;
    }

    if (grid_x * shape[i] < UINT32_MAX) {
      grid_x *= shape[i];
    } else {
      grid_y *= shape[i];
    }

    if (num_threads > 1) {
      if (grid_x % num_threads == 0) {
        grid_x /= num_threads;
        num_threads = 1;
      } else if (grid_y % num_threads == 0) {
        grid_y /= num_threads;
        num_threads = 1;
      }
    }
  }
  if (grid_y > UINT32_MAX || grid_x > UINT32_MAX || num_threads > 1) {
    throw std::runtime_error("Unable to safely factor shape.");
  }
  if (grid_y > grid_x) {
    std::swap(grid_x, grid_y);
  }
  return {static_cast<uint32_t>(grid_x), static_cast<uint32_t>(grid_y), 1};
}

std::string get_primitive_string(Primitive* primitive) {
  std::ostringstream op_t;
  primitive->print(op_t);
  return op_t.str();
}

} // namespace mlx::core
