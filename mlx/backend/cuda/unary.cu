// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernels/unary_ops.cuh"
#include "mlx/primitives.h"

#include <thrust/device_ptr.h>
#include <thrust/transform.h>

namespace mlx::core {

namespace {

template <typename Op, typename In, typename Out>
constexpr bool is_supported_unary_op() {
  if (std::is_same_v<Op, mxcuda::Abs> || std::is_same_v<Op, mxcuda::Negative> ||
      std::is_same_v<Op, mxcuda::Sign>) {
    return std::is_same_v<In, Out>;
  }
  if (std::is_same_v<Op, mxcuda::ArcCos> ||
      std::is_same_v<Op, mxcuda::ArcCosh> ||
      std::is_same_v<Op, mxcuda::ArcSin> ||
      std::is_same_v<Op, mxcuda::ArcSinh> ||
      std::is_same_v<Op, mxcuda::ArcTan> ||
      std::is_same_v<Op, mxcuda::ArcTanh> || std::is_same_v<Op, mxcuda::Erf> ||
      std::is_same_v<Op, mxcuda::ErfInv> || std::is_same_v<Op, mxcuda::Expm1> ||
      std::is_same_v<Op, mxcuda::Log1p> || std::is_same_v<Op, mxcuda::Log> ||
      std::is_same_v<Op, mxcuda::Log2> || std::is_same_v<Op, mxcuda::Log10> ||
      std::is_same_v<Op, mxcuda::Sigmoid> || std::is_same_v<Op, mxcuda::Sqrt> ||
      std::is_same_v<Op, mxcuda::Rsqrt>) {
    return std::is_same_v<In, Out> && is_floating_v<In>;
  }
  if (std::is_same_v<Op, mxcuda::BitwiseInvert>) {
    return std::is_same_v<In, Out> && std::is_integral_v<In> &&
        !std::is_same_v<In, bool>;
  }
  if (std::is_same_v<Op, mxcuda::Ceil> || std::is_same_v<Op, mxcuda::Floor> ||
      std::is_same_v<Op, mxcuda::Square>) {
    return std::is_same_v<In, Out> && !std::is_same_v<In, complex64_t>;
  }
  if (std::is_same_v<Op, mxcuda::Conjugate>) {
    return std::is_same_v<In, Out> && std::is_same_v<In, complex64_t>;
  }
  if (std::is_same_v<Op, mxcuda::Cos> || std::is_same_v<Op, mxcuda::Cosh> ||
      std::is_same_v<Op, mxcuda::Exp> || std::is_same_v<Op, mxcuda::Round> ||
      std::is_same_v<Op, mxcuda::Sin> || std::is_same_v<Op, mxcuda::Sinh> ||
      std::is_same_v<Op, mxcuda::Tan> || std::is_same_v<Op, mxcuda::Tanh>) {
    return std::is_same_v<In, Out> &&
        (is_floating_v<In> || std::is_same_v<In, complex64_t>);
  }
  if (std::is_same_v<Op, mxcuda::Imag> || std::is_same_v<Op, mxcuda::Real>) {
    return std::is_same_v<In, complex64_t> && std::is_same_v<Out, float>;
  }
  if (std::is_same_v<Op, mxcuda::LogicalNot>) {
    return std::is_same_v<In, Out> && std::is_same_v<In, bool>;
  }
  return false;
}

template <typename Op>
void unary_op_gpu_inplace(
    const std::vector<array>& inputs,
    array& out,
    const std::string& op,
    const Stream& s) {
  auto& in = inputs[0];
  bool contig = in.flags().contiguous;
  if (in.size() == 0) {
    return;
  }

  auto maybe_collapse = [contig, &in, &out]() {
    if (!contig) {
      return collapse_contiguous_dims(in);
    } else {
      return std::make_pair(Shape{}, Strides{});
    }
  };
  auto [shape, strides] = maybe_collapse();
  int ndim = shape.size();
  size_t nthreads = contig ? in.data_size() : in.size();
  bool large;
  if (!contig) {
    large = in.data_size() > INT32_MAX || out.size() > INT32_MAX;
  } else {
    large = in.data_size() > UINT32_MAX;
  }
  int work_per_thread = !contig && large ? 4 : 1;

  std::ignore = work_per_thread;

  auto& encoder = mxcuda::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  if (!contig) {
    throw std::runtime_error(fmt::format(
        "General unary op {} not implemented for CUDA backend.", op));
  } else {
    encoder.launch_thrust([&](auto policy) {
      MLX_SWITCH_ALL_TYPES(in.dtype(), CTYPE_IN, [&]() {
        MLX_SWITCH_ALL_TYPES(out.dtype(), CTYPE_OUT, [&]() {
          if constexpr (is_supported_unary_op<Op, CTYPE_IN, CTYPE_OUT>()) {
            using InType = cuda_type_t<CTYPE_IN>;
            using OutType = cuda_type_t<CTYPE_OUT>;
            thrust::transform(
                policy,
                thrust::device_pointer_cast(in.data<InType>()),
                thrust::device_pointer_cast(in.data<InType>() + in.data_size()),
                thrust::device_pointer_cast(out.data<OutType>()),
                Op());
          } else {
            throw std::runtime_error(fmt::format(
                "Can not do unary op {} on input of {} with output of {}.",
                op,
                dtype_to_string(in.dtype()),
                dtype_to_string(out.dtype())));
          }
        });
      });
    });
  }
}

template <typename Op>
void unary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const std::string& op,
    const Stream& s) {
  auto& in = inputs[0];
  bool contig = in.flags().contiguous;
  if (contig) {
    if (in.is_donatable() && in.itemsize() == out.itemsize()) {
      out.copy_shared_buffer(in);
    } else {
      out.set_data(
          allocator::malloc(in.data_size() * out.itemsize()),
          in.data_size(),
          in.strides(),
          in.flags());
    }
  } else {
    out.set_data(allocator::malloc(out.nbytes()));
  }
  unary_op_gpu_inplace<Op>(inputs, out, op, s);
}

} // namespace

#define UNARY_GPU(func)                                         \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) {       \
    auto& s = out.primitive().stream();                                     \
    unary_op_gpu<mxcuda::func>(inputs, out, get_primitive_string(this), s); \
  }

UNARY_GPU(Abs)
UNARY_GPU(ArcCos)
UNARY_GPU(ArcCosh)
UNARY_GPU(ArcSin)
UNARY_GPU(ArcSinh)
UNARY_GPU(ArcTan)
UNARY_GPU(ArcTanh)
UNARY_GPU(BitwiseInvert)
UNARY_GPU(Ceil)
UNARY_GPU(Conjugate)
UNARY_GPU(Cos)
UNARY_GPU(Cosh)
UNARY_GPU(Erf)
UNARY_GPU(ErfInv)
UNARY_GPU(Exp)
UNARY_GPU(Expm1)
UNARY_GPU(Floor)
UNARY_GPU(Imag)
UNARY_GPU(Log1p)
UNARY_GPU(LogicalNot)
UNARY_GPU(Negative)
UNARY_GPU(Real)
UNARY_GPU(Sigmoid)
UNARY_GPU(Sign)
UNARY_GPU(Sin)
UNARY_GPU(Sinh)
UNARY_GPU(Square)
UNARY_GPU(Tan)
UNARY_GPU(Tanh)

void Log::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = out.primitive().stream();
  auto op = get_primitive_string(this);
  switch (base_) {
    case Base::e:
      unary_op_gpu<mxcuda::Log>(inputs, out, op, s);
      break;
    case Base::two:
      unary_op_gpu<mxcuda::Log2>(inputs, out, op, s);
      break;
    case Base::ten:
      unary_op_gpu<mxcuda::Log10>(inputs, out, op, s);
      break;
  }
}

void Round::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  auto& s = out.primitive().stream();
  if (issubdtype(in.dtype(), inexact)) {
    unary_op_gpu<mxcuda::Round>(inputs, out, get_primitive_string(this), s);
  } else {
    // No-op integer types
    out.copy_shared_buffer(in);
  }
}

void Sqrt::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = out.primitive().stream();
  if (recip_) {
    unary_op_gpu<mxcuda::Rsqrt>(inputs, out, "Rsqrt", s);
  } else {
    unary_op_gpu<mxcuda::Sqrt>(inputs, out, "Sqrt", s);
  }
}

} // namespace mlx::core
