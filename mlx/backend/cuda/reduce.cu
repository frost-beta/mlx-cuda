// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/reduce.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernels/reduce_ops.cuh"
#include "mlx/backend/metal/copy.h"
#include "mlx/primitives.h"

#include <assert.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/constant_iterator.h>
#include <cub/device/device_reduce.cuh>

namespace mlx::core {

namespace {

#define MLX_FORALL_REDUCE_TYPES(_, ...) \
  _(And, __VA_ARGS__)                   \
  _(Or, __VA_ARGS__)                    \
  _(Sum, __VA_ARGS__)                   \
  _(Prod, __VA_ARGS__)                  \
  _(Max, __VA_ARGS__)                   \
  _(Min, __VA_ARGS__)

#define MLX_SWITCH_CASE_REDUCE_TYPE(TYPE, OP, ...) \
  case Reduce::TYPE: {                             \
    using OP = mxcuda::TYPE;                       \
    __VA_ARGS__;                                   \
    break;                                         \
  }

#define MLX_SWITCH_REDUCE_TYPES(TYPE, OP, ...)                            \
  switch (TYPE) {                                                         \
    MLX_FORALL_REDUCE_TYPES(MLX_SWITCH_CASE_REDUCE_TYPE, OP, __VA_ARGS__) \
  }

template <typename Op, typename T>
constexpr bool is_supported_reduce_op() {
  if (std::is_same_v<Op, mxcuda::And> || std::is_same_v<Op, mxcuda::Or>) {
    return std::is_same_v<T, bool>;
  }
  if (std::is_same_v<Op, mxcuda::Sum> || std::is_same_v<Op, mxcuda::Prod>) {
    return !std::is_same_v<T, bool>;
  }
  if (std::is_same_v<Op, mxcuda::Min> || std::is_same_v<Op, mxcuda::Max>) {
    return true;
  }
  return false;
}

template <typename... Args>
void all_reduce(mxcuda::CommandEncoder& encoder, Args&&... args) {
  // Get required size for temporary storage and allocate it.
  size_t size;
  CHECK_CUDA_ERROR(cub::DeviceReduce::Reduce(nullptr, size, args...));
  array temp(allocator::malloc(size), {static_cast<int>(size)}, uint8);
  encoder.add_temporary(temp);
  // Actually run reduce.
  CHECK_CUDA_ERROR(cub::DeviceReduce::Reduce(temp.data<void>(), size, args...));
}

} // namespace

void Reduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  array in = inputs[0];

  // Make sure no identity reductions trickle down here
  assert(!axes_.empty());
  assert(out.size() != in.size());

  out.set_data(allocator::malloc(out.nbytes()));

  auto& s = stream();
  auto& encoder = mxcuda::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);

  // Fill out with init value.
  if (in.size() == 0) {
    encoder.launch_thrust([&](auto policy) {
      MLX_SWITCH_ALL_TYPES(in.dtype(), CTYPE, [&]() {
        MLX_SWITCH_REDUCE_TYPES(reduce_type_, OP, {
          if constexpr (is_supported_reduce_op<OP, CTYPE>()) {
            using InType = cuda_type_t<CTYPE>;
            using OutType = mxcuda::ReduceInit<OP, InType>::type;
            thrust::copy_n(
                policy,
                thrust::make_constant_iterator(
                    mxcuda::ReduceInit<OP, InType>::value),
                out.data_size(),
                thrust::device_pointer_cast(out.data<OutType>()));
          } else {
            throw std::runtime_error(fmt::format(
                "Can not do reduce init op on dtype {}.",
                dtype_to_string(in.dtype())));
          }
        });
      });
    });
    return;
  }

  // Reduce.
  ReductionPlan plan = get_reduction_plan(in, axes_);

  // If it is a general reduce then copy the input to a contiguous array and
  // recompute the plan.
  if (plan.type == GeneralReduce) {
    array in_copy(in.shape(), in.dtype(), nullptr, {});
    copy_gpu(in, in_copy, CopyType::General, s);
    encoder.add_temporary(in_copy);
    in = in_copy;
    plan = get_reduction_plan(in, axes_);
  }

  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_ALL_TYPES(in.dtype(), CTYPE, [&]() {
      MLX_SWITCH_REDUCE_TYPES(reduce_type_, OP, {
        if constexpr (is_supported_reduce_op<OP, CTYPE>()) {
          using InType = cuda_type_t<CTYPE>;
          using OutType = mxcuda::ReduceInit<OP, InType>::type;
          if (plan.type == ContiguousAllReduce) {
            all_reduce(
                encoder,
                thrust::device_pointer_cast(in.data<InType>()),
                thrust::device_pointer_cast(out.data<OutType>()),
                in.data_size(),
                OP(),
                mxcuda::ReduceInit<OP, InType>::value,
                stream);
          } else if (
              plan.type == ContiguousReduce ||
              plan.type == GeneralContiguousReduce) {
            throw std::runtime_error("Reduce not implemented in CUDA backend.");
          } else if (
              plan.type == ContiguousStridedReduce ||
              plan.type == GeneralStridedReduce) {
            throw std::runtime_error("Reduce not implemented in CUDA backend.");
          }
        } else {
          throw std::runtime_error(fmt::format(
              "Can not do reduce op on dtype {}.",
              dtype_to_string(in.dtype())));
        }
      });
    });
  });
}

} // namespace mlx::core
