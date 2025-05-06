// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/dtype_utils.cuh"
#include "mlx/backend/cuda/kernels/utils.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/fast_primitives.h"

#include <cooperative_groups.h>
#include <nvtx3/nvtx3.hpp>
#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

inline __device__ float2 plus(const float2& a, const float2& b) {
  return float2{a.x + b.x, a.y + b.y};
}

template <typename T, uint32_t BLOCK_DIM, uint32_t N_READS = 4>
__global__ void layer_norm(
    const T* x,
    const T* w,
    const T* b,
    T* out,
    const float eps,
    const uint32_t axis_size,
    const uint32_t w_stride,
    const uint32_t b_stride) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();

  x += grid.block_rank() * axis_size;
  out += grid.block_rank() * axis_size;

  float2 sum{0, 0};
  for (int r = 0; r < cuda::ceil_div(axis_size, BLOCK_DIM * N_READS); r++) {
    float2 vals[N_READS] = {0};
    cub::LoadDirectBlocked(
        r * BLOCK_DIM + block.thread_rank(),
        thrust::make_transform_iterator(
            x,
            [] __device__(T i) {
              float f = static_cast<float>(i);
              return float2{f, f * f};
            }),
        vals,
        axis_size);
    sum = plus(sum, cub::ThreadReduce(vals, plus));
  }

  using BlockReduceT = cub::BlockReduce<float2, BLOCK_DIM>;
  __shared__ typename BlockReduceT::TempStorage temp;
  sum = BlockReduceT(temp).Reduce(sum, plus);

  __shared__ float2 mean_normalizer;
  if (block.thread_rank() == 0) {
    float mean = sum.x / axis_size;
    float variance = sum.y / axis_size - mean * mean;
    float normalizer = rsqrt(variance + eps);
    mean_normalizer = {mean, normalizer};
  }
  block.sync();

  float mean = mean_normalizer.x;
  float normalizer = mean_normalizer.y;

  for (int r = 0; r < cuda::ceil_div(axis_size, BLOCK_DIM * N_READS); r++) {
    auto index = r * BLOCK_DIM + block.thread_rank();
    T vals[N_READS];
    cub::LoadDirectBlocked(index, x, vals, axis_size);
    for (int i = 0; i < N_READS; i++) {
      float norm = (static_cast<float>(vals[i]) - mean) * normalizer;
      T wi = w[w_stride * (index * N_READS + i)];
      T bi = b[b_stride * (index * N_READS + i)];
      vals[i] = wi * static_cast<T>(norm) + bi;
    }
    cub::StoreDirectBlocked(index, out, vals, axis_size);
  }
}

} // namespace cu

namespace fast {

void LayerNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("LayerNorm::eval_gpu");
  auto& s = stream();
  auto& out = outputs[0];

  // Make sure that the last dimension is contiguous.
  auto set_output = [&s, &out](const array& x) {
    bool no_copy = x.flags().contiguous && x.strides()[x.ndim() - 1] == 1;
    if (no_copy && x.ndim() > 1) {
      auto s = x.strides()[x.ndim() - 2];
      no_copy &= (s == 0 || s == x.shape().back());
    }
    if (no_copy) {
      if (x.is_donatable()) {
        out.copy_shared_buffer(x);
      } else {
        out.set_data(
            allocator::malloc(x.data_size() * x.itemsize()),
            x.data_size(),
            x.strides(),
            x.flags());
      }
      return x;
    } else {
      auto x_copy = array(x.shape(), x.dtype(), nullptr, {});
      copy_gpu(x, x_copy, CopyType::General, s);
      out.copy_shared_buffer(x_copy);
      return x_copy;
    }
  };

  array o = set_output(inputs[0]);
  const array& x = o.data_shared_ptr() ? o : out;
  const array& w = inputs[1];
  const array& b = inputs[2];

  uint32_t axis_size = x.shape().back();
  uint32_t n_rows = x.data_size() / axis_size;
  uint32_t w_stride = (w.ndim() == 1) ? w.strides()[0] : 0;
  uint32_t b_stride = (b.ndim() == 1) ? b.strides()[0] : 0;

  auto& encoder = cu::get_command_encoder(s);
  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_input_array(b);
  encoder.set_output_array(out);
  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_FLOAT_TYPES_CHECKED(out.dtype(), "layernorm", CTYPE, {
      using DataType = cuda_type_t<CTYPE>;
      constexpr uint32_t N_READS = 4;
      MLX_SWITCH_BLOCK_DIM(cuda::ceil_div(axis_size, N_READS), BLOCK_DIM, {
        auto kernel = cu::layer_norm<DataType, BLOCK_DIM, N_READS>;
        kernel<<<n_rows, BLOCK_DIM, 0, stream>>>(
            x.data<DataType>(),
            w.data<DataType>(),
            b.data<DataType>(),
            out.data<DataType>(),
            eps_,
            axis_size,
            w_stride,
            b_stride);
      });
    });
  });
}

} // namespace fast

} // namespace mlx::core
