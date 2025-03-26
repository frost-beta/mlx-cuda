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
        thrust::copy_n(
            policy,
            thrust::make_constant_iterator(mxcuda::Sum<CTYPE>::init),
            out.size(),
            thrust::device_pointer_cast(out.data<CTYPE>()));
      });
    });
    throw std::runtime_error("Reduce plan not implemented in CUDA.");
  }
}

} // namespace mlx::core
