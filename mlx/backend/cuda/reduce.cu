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

namespace mlx::core {

namespace {

#define MLX_FORALL_REDUCE_TYPES(_, ...) \
  _(And, __VA_ARGS__)                   \
  _(Or, __VA_ARGS__)                    \
  _(Sum, __VA_ARGS__)                   \
  _(Prod, __VA_ARGS__)                  \
  _(Max, __VA_ARGS__)                   \
  _(Min, __VA_ARGS__)

#define MLX_SWITCH_CASE_REDUCE_TYPE(TYPE, CTYPE, OP, ...) \
  case Reduce::TYPE: {                                    \
    using OP = mxcuda::TYPE<CTYPE>;                       \
    __VA_ARGS__;                                          \
    break;                                                \
  }

#define MLX_SWITCH_REDUCE_TYPES(TYPE, CTYPE, OP, ...)        \
  switch (TYPE) {                                            \
    MLX_FORALL_REDUCE_TYPES(                                 \
        MLX_SWITCH_CASE_REDUCE_TYPE, CTYPE, OP, __VA_ARGS__) \
  }

template <template<typename> class Op, typename T>
constexpr bool is_supported_reduce_init(Op<T>) {
  if (std::is_same_v<Op<T>, mxcuda::And<T>> ||
      std::is_same_v<Op<T>, mxcuda::Or<T>>) {
    return std::is_same_v<T, bool>;
  }
  if (std::is_same_v<Op<T>, mxcuda::Sum<T>> ||
      std::is_same_v<Op<T>, mxcuda::Prod<T>>) {
    return !std::is_same_v<T, bool>;
  }
  if (std::is_same_v<Op<T>, mxcuda::Min<T>> ||
      std::is_same_v<Op<T>, mxcuda::Max<T>>) {
    return true;
  }
  return false;
}

} // namespace

void Reduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  array in = inputs[0];

  // Make sure no identity reductions trickle down here
  assert(!axes_.empty());
  assert(out.size() != in.size());

  // Continue with reduction operation
  // Minimum of 4 bytes since we use size 4 structs for all reduce
  // and metal will complain o/w
  size_t min_bytes = std::max(out.nbytes(), 4ul);
  out.set_data(allocator::malloc(min_bytes));

  auto& s = stream();
  auto& encoder = mxcuda::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);

  // Reduce
  if (in.size() > 0) {
    ReductionPlan plan = get_reduction_plan(in, axes_);

    // If it is a general reduce then copy the input to a contiguous array and
    // recompute the plan.
    //
    // TODO: This can be avoided by making the output have the same strides as
    //       input for the axes with stride smaller than the minimum reduction
    //       stride.
    if (plan.type == GeneralReduce) {
      array in_copy(in.shape(), in.dtype(), nullptr, {});
      copy_gpu(in, in_copy, CopyType::General, s);
      encoder.add_temporary(in_copy);
      in = in_copy;
      plan = get_reduction_plan(in, axes_);
    }

    throw std::runtime_error("Reduce plan not implemented in CUDA.");
  } else {
    encoder.launch_thrust([&](auto policy) {
      MLX_SWITCH_CUDA_TYPES(out.dtype(), CTYPE, [&]() {
        MLX_SWITCH_REDUCE_TYPES(reduce_type_, CTYPE, OP, {
          if constexpr (is_supported_reduce_init(OP{})) {
            thrust::copy_n(
                policy,
                thrust::make_constant_iterator(OP::init),
                out.size(),
                thrust::device_pointer_cast(out.data<CTYPE>()));
          } else {
            throw std::runtime_error(fmt::format(
                "Can not do reduce init op on dtype {}.",
                dtype_to_string(out.dtype())));
          }
        });
      });
    });
  }
}

} // namespace mlx::core
