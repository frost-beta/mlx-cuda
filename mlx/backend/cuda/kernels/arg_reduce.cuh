// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/kernels/utils.cuh"

#include <cooperative_groups.h>

namespace mlx::core::mxcuda {

namespace cg = cooperative_groups;

template <typename U>
struct IndexValPair {
  uint32_t index;
  U val;
};

template <typename U>
struct ArgMin {
  static constexpr U init = Limits<U>::max;

  __forceinline__ __device__ IndexValPair<U> reduce(
      IndexValPair<U> best,
      IndexValPair<U> current) {
    if (best.val > current.val ||
        (best.val == current.val && best.index > current.index)) {
      return current;
    } else {
      return best;
    }
  }

  template <int N>
  __forceinline__ __device__ IndexValPair<U>
  reduce_many(IndexValPair<U> best, U* vals, uint32_t offset) {
    for (int i = 0; i < N; i++) {
      if (vals[i] < best.val) {
        best.val = vals[i];
        best.index = offset + i;
      }
    }
    return best;
  }
};

template <typename U>
struct ArgMax {
  static constexpr U init = Limits<U>::min;

  __forceinline__ __device__ IndexValPair<U> reduce(
      IndexValPair<U> best,
      IndexValPair<U> current) {
    if (best.val < current.val ||
        (best.val == current.val && best.index > current.index)) {
      return current;
    } else {
      return best;
    }
  }

  template <int N>
  __forceinline__ __device__ IndexValPair<U>
  reduce_many(IndexValPair<U> best, U* vals, uint32_t offset) {
    for (int i = 0; i < N; i++) {
      if (vals[i] > best.val) {
        best.val = vals[i];
        best.index = offset + i;
      }
    }
    return best;
  }
};

template <typename U>
__device__ IndexValPair<U> warp_shuffle_down(
    const cg::thread_block_tile<WARP_SIZE>& g,
    IndexValPair<U> data,
    uint16_t delta) {
  return {g.shfl_down(data.index, delta), g.shfl_down(data.val, delta)};
}

template <typename T, typename Op, int N_READS = 4>
__global__ void arg_reduce_general(
    const T* in,
    uint32_t* out,
    const __grid_constant__ Shape shape,
    const __grid_constant__ Strides in_strides,
    const __grid_constant__ Strides out_strides,
    size_t ndim,
    int64_t axis_stride,
    size_t axis_size) {
  // Shapes and strides *do not* contain the reduction axis. The reduction size
  // and stride are provided in axis_stride and axis_size.
  //
  // Note: in shape == out shape with this convention.
  //
  // The sketch of the kernel is as follows.
  //    1. Launch prod(shape) * blockDim.x threads.
  //    2. Loop ceildiv(axis_size / blockDim.x) times
  //    3. Read input values
  //    4. Reduce among them and go to 3
  //    5. Reduce in each warp
  //    6. Write in the shared memory
  //    7. Reduce them across block
  //    8. Write the output without need for atomic
  Op op;
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  // Compute the input/output index. There is one beginning and one output for
  // the whole threadgroup.
  auto in_idx = elem_to_loc(blockIdx.x, shape.data(), in_strides.data(), ndim);
  auto out_idx =
      elem_to_loc(blockIdx.x, shape.data(), out_strides.data(), ndim);

  IndexValPair<T> best{0, Op::init};

  __shared__ IndexValPair<T> local_data[MAX_BLOCK_DIM / WARP_SIZE];

  // Loop over the reduction axis in blockDim.x * N_READS buckets.
  uint32_t block_size = N_READS * blockDim.x;
  for (uint32_t r = 0; r < ceil_div(axis_size, block_size); r++) {
    // Read the current value.
    uint32_t current_index = r * block_size + threadIdx.x * N_READS;
    uint32_t offset = current_index;
    const T* current_in = in + in_idx + current_index * axis_stride;
    T vals[N_READS];
    for (int i = 0; i < N_READS; i++) {
      vals[i] = (current_index < axis_size) ? *current_in : T(Op::init);
      current_index++;
      current_in += axis_stride;
    }
    best = op.template reduce_many<N_READS>(best, vals, offset);
  }
  // At this point we have reduced the axis into thread group best values so we
  // need to reduce across the thread group.

  // First per warp reduction.
  for (int delta = warp.size() / 2; delta > 0; delta /= 2) {
    IndexValPair<T> neighbor = warp_shuffle_down(warp, best, delta);
    best = op.reduce(best, neighbor);
  }

  // Write to the shared memory.
  if (warp.thread_rank() == 0) {
    local_data[warp.meta_group_rank()] = best;
  }
  block.sync();
  if (warp.meta_group_rank() != 0) {
    return;
  }

  // Read the appropriate value from local data and perform one warp reduction.
  if (warp.thread_rank() < warp.meta_group_size()) {
    best = local_data[warp.thread_rank()];
  }
  for (int delta = warp.meta_group_size() / 2; delta > 0; delta /= 2) {
    IndexValPair<T> neighbor = warp_shuffle_down(warp, best, delta);
    best = op.reduce(best, neighbor);
  }

  // Finally write the output.
  if (block.thread_rank() == 0) {
    out[out_idx] = best.index;
  }
}

} // namespace mlx::core::mxcuda
