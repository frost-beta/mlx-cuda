// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/metal/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <assert.h>

namespace mlx::core {

void copy_gpu_inplace(
    const array& in,
    array& out,
    const Shape& data_shape,
    const Strides& strides_in_pre,
    const Strides& strides_out_pre,
    int64_t inp_offset,
    int64_t out_offset,
    CopyType ctype,
    const Stream& s,
    const std::optional<array>& dynamic_i_offset /* = std::nullopt */,
    const std::optional<array>& dynamic_o_offset /* = std::nullopt */) {
  if (out.size() == 0) {
    return;
  }
  // Try to collapse contiguous dims
  auto maybe_collapse =
      [ctype, &data_shape, &strides_in_pre, &strides_out_pre]() {
        if (ctype == CopyType::General || ctype == CopyType::GeneralGeneral) {
          auto [shape, strides] = collapse_contiguous_dims(
              data_shape,
              std::vector{strides_in_pre, strides_out_pre},
              /* size_cap = */ INT32_MAX);
          return std::make_tuple(shape, strides[0], strides[1]);
        } else {
          Strides e{};
          return std::make_tuple(Shape{}, e, e);
        }
      };
  auto [shape, strides_in_, strides_out_] = maybe_collapse();
  int ndim = shape.size();
  bool large;
  if (ctype == CopyType::General || ctype == CopyType::GeneralGeneral) {
    // Allow for negative strides
    large = in.data_size() > INT32_MAX || out.data_size() > INT32_MAX;
  } else {
    large = out.data_size() > UINT32_MAX;
  }
  bool dynamic = dynamic_i_offset || dynamic_o_offset;

  bool donate_in = in.data_shared_ptr() == nullptr;
  const array& input = donate_in ? out : in;

  auto& encoder = mxcuda::get_command_encoder(s);
  encoder.set_input_array(input);
  encoder.set_output_array(out);
  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_ALL_TYPES(input.dtype(), "copy in", CTYPE_IN, [&]() {
      MLX_SWITCH_ALL_TYPES(out.dtype(), "copy out", CTYPE_OUT, [&]() {
        if (ctype == CopyType::General || ctype == CopyType::GeneralGeneral) {
          throw std::runtime_error(
              "General copy not implemented for CUDA backend.");
        } else {
        }
      });
    });
  });
}

// TODO: Code below are identical to backend/metal/copy.cpp.
void copy_gpu(const array& in, array& out, CopyType ctype, const Stream& s) {
  bool donated = set_copy_output_data(in, out, ctype);
  if (donated && in.dtype() == out.dtype()) {
    // If the output has the same type as the input then there is nothing to
    // copy, just use the buffer.
    return;
  }
  if (ctype == CopyType::GeneralGeneral) {
    ctype = CopyType::General;
  }
  copy_gpu_inplace(in, out, ctype, s);
}

void copy_gpu(const array& in, array& out, CopyType ctype) {
  copy_gpu(in, out, ctype, out.primitive().stream());
}

void copy_gpu_inplace(
    const array& in,
    array& out,
    CopyType ctype,
    const Stream& s) {
  assert(in.shape() == out.shape());
  return copy_gpu_inplace(
      in, out, in.shape(), in.strides(), out.strides(), 0, 0, ctype, s);
}

void copy_gpu_inplace(
    const array& in,
    array& out,
    const Strides& i_strides,
    int64_t i_offset,
    CopyType ctype,
    const Stream& s) {
  assert(in.shape() == out.shape());
  return copy_gpu_inplace(
      in, out, in.shape(), i_strides, out.strides(), i_offset, 0, ctype, s);
}

} // namespace mlx::core
