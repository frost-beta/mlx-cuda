// Copyright © 2025 Apple Inc.

namespace mlx::core::mxcuda {

template <typename T, typename U>
__global__ void copy_s(const T* src, U* dst, uint32_t size) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < size) {
    dst[index] = static_cast<U>(src[0]);
  }
}

template <typename T, typename U>
__global__ void copy_v(const T* src, U* dst, uint32_t size) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < size) {
    dst[index] = static_cast<U>(src[index]);
  }
}

} // namespace mlx::core::mxcuda
