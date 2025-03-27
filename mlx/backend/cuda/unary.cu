// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernels/unary_ops.cuh"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

template <typename T>
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

  throw std::runtime_error("Unary op not implemented for CUDA backend.");
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

#define UNARY_GPU(func)                                                     \
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
UNARY_GPU(Conjugate)
UNARY_GPU(Cos)
UNARY_GPU(Cosh)
UNARY_GPU(Erf)
UNARY_GPU(ErfInv)
UNARY_GPU(Exp)
UNARY_GPU(Expm1)
UNARY_GPU(Imag)
UNARY_GPU(Log1p)
UNARY_GPU(LogicalNot)
UNARY_GPU(Floor)
UNARY_GPU(Ceil)
UNARY_GPU(Negative)
UNARY_GPU(Real)
UNARY_GPU(Sigmoid)
UNARY_GPU(Sign)
UNARY_GPU(Sin)
UNARY_GPU(Sinh)
UNARY_GPU(Square)
UNARY_GPU(Sqrt)
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

} // namespace mlx::core
