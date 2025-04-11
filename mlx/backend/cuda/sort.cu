// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernels/utils.cuh"
#include "mlx/dtype_utils.h"
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

void Sort::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Sort::eval_gpu");
  assert(inputs.size() == 1);
  auto& in = inputs[0];

  out.set_data(allocator::malloc(out.nbytes()));

  auto& s = stream();
  auto& encoder = mxcuda::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);

  int axis = axis_ < 0 ? axis_ + in.ndim() : axis_;
  int nsort = in.shape(axis);
  int nsegments = in.data_size() / nsort;

  if (!in.flags().contiguous || in.strides()[axis] != 1) {
    throw std::runtime_error(
        "Can only sort the innermost axis of contiguous array");
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
}

} // namespace mlx::core
