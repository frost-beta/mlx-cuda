// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/kernels/utils.cuh"

#include <cooperative_groups.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>

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

  __device__ IndexValPair<U> operator()(
      const IndexValPair<U>& best,
      const IndexValPair<U>& current) {
    if (best.val > current.val ||
        (best.val == current.val && best.index > current.index)) {
      return current;
    } else {
      return best;
    }
  }

  template <int N>
  __device__ IndexValPair<U>
  reduce_many(IndexValPair<U> best, U* vals, uint32_t offset) {
    CUDA_UNROLL for (int i = 0; i < N; i++) {
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

  __device__ IndexValPair<U> operator()(
      const IndexValPair<U>& best,
      const IndexValPair<U>& current) {
    if (best.val < current.val ||
        (best.val == current.val && best.index > current.index)) {
      return current;
    } else {
      return best;
    }
  }

  template <int N>
  __device__ IndexValPair<U>
  reduce_many(IndexValPair<U> best, U* vals, uint32_t offset) {
    CUDA_UNROLL for (int i = 0; i < N; i++) {
      if (vals[i] > best.val) {
        best.val = vals[i];
        best.index = offset + i;
      }
    }
    return best;
  }
};

template <typename U>
inline __device__ IndexValPair<U> warp_shuffle_down(
    const cg::thread_block_tile<WARP_SIZE>& g,
    const IndexValPair<U>& data,
    int delta) {
  return {g.shfl_down(data.index, delta), g.shfl_down(data.val, delta)};
}

template <typename T, typename Op, int BLOCK_DIM, int N_READS = 4>
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
  Op op;

  // Compute the input/output index. There is one beginning and one output for
  // the whole threadgroup.
  auto elem = cg::this_grid().block_rank();
  auto in_idx = elem_to_loc(elem, shape.data(), in_strides.data(), ndim);
  auto out_idx = elem_to_loc(elem, shape.data(), out_strides.data(), ndim);

  IndexValPair<T> best{0, Op::init};

  // Loop over the reduction axis in N_READS * block.size() buckets.
  auto block = cg::this_thread_block();
  uint32_t block_size = N_READS * block.size();
  for (uint32_t r = 0; r < ceil_div(axis_size, block_size); r++) {
    // Read the current value.
    uint32_t current_index = r * block_size + block.thread_rank() * N_READS;
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

  typedef cub::BlockReduce<IndexValPair<T>, BLOCK_DIM> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp;

  best = BlockReduceT(temp).Reduce(best, op);

  if (block.thread_rank() == 0) {
    out[out_idx] = best.index;
  }
}

} // namespace mlx::core::mxcuda
