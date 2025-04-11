// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernels/utils.cuh"
#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/metal_impl.h"
#include "mlx/dtype_utils.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

#include <assert.h>
#include <nvtx3/nvtx3.hpp>
#include <cub/device/device_segmented_sort.cuh>

namespace mlx::core {

namespace {

struct OffsetIterator {
  int stride;
  int begin;
  __device__ int operator[](int i) const {
    return stride * (begin + i);
  }
};

template <typename... Args>
void segmented_sort(mxcuda::CommandEncoder& encoder, Args&&... args) {
  // Allocate temporary storage.
  size_t size;
  CHECK_CUDA_ERROR(
      cub::DeviceSegmentedSort::StableSortKeys(nullptr, size, args...));
  array temp(allocator::malloc(size), {static_cast<int>(size)}, uint8);
  encoder.add_temporary(temp);
  // Run op.
  CHECK_CUDA_ERROR(cub::DeviceSegmentedSort::StableSortKeys(
      temp.data<void>(), size, args...));
}

} // namespace

void Sort::eval_gpu(const std::vector<array>& inputs, array& out_) {
  nvtx3::scoped_range r("Sort::eval_gpu");
  assert(inputs.size() == 1);
  array in = inputs[0];
  array out = out_;

  auto& s = stream();
  auto& encoder = mxcuda::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out_);

  int axis = axis_ < 0 ? axis_ + in.ndim() : axis_;
  int nsort = in.shape(axis);
  int nsegments = in.data_size() / nsort;

  // If we are not sorting the innermost dimension of a contiguous array,
  // transpose and make a copy.
  bool is_segmented_sort = in.flags().contiguous && in.strides()[axis] == 1;
  if (!is_segmented_sort) {
    array trans = swapaxes(in, axis, -1, s);
    metal::eval(trans);
    in = array(trans.shape(), trans.dtype(), nullptr, {});
    copy_gpu(trans, in, CopyType::General, s);
    encoder.add_temporary(in);
    out = array(allocator::malloc(out.nbytes()), in.shape(), in.dtype());
    encoder.add_temporary(out);
  } else {
    out.set_data(allocator::malloc(out.nbytes()));
  }

  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_ALL_TYPES(in.dtype(), CTYPE, [&]() {
      if constexpr (!std::is_same_v<CTYPE, complex64_t>) {
        using Type = cuda_type_t<CTYPE>;
        segmented_sort(
            encoder,
            in.data<Type>(),
            out.data<Type>(),
            in.data_size(),
            nsegments,
            OffsetIterator{nsort, 0},
            OffsetIterator{nsort, 1},
            stream);
      } else {
        throw std::runtime_error(
            "sort does not support complex numbers in CUDA backend");
      }
    });
  });

  if (!is_segmented_sort) {
    // Swap the sorted axis back.
    // TODO: Do in-place transpose instead of using a temporary out array.
    array trans = swapaxes(out, axis, -1);
    metal::eval(trans);
    copy_gpu(trans, out_, CopyType::General, s);
  }
}

} // namespace mlx::core
