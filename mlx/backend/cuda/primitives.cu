// Copyright © 2023-2024 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernels/arg_reduce.cuh"
#include "mlx/backend/metal/copy.h"
#include "mlx/distributed/primitives.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"

#include <assert.h>

#define NO_GPU_MULTI(func)                                             \
  void func::eval_gpu(                                                 \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    throw std::runtime_error(#func " has no GPU implementation.");     \
  }

#define NO_GPU(func)                                                  \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    throw std::runtime_error(#func " has no GPU implementation.");    \
  }

namespace mlx::core {

void ArgReduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  out.set_data(allocator::malloc(out.nbytes()));
  auto& s = stream();

  // Prepare the shapes, strides and axis arguments.
  auto in_strides = in.strides();
  auto shape = in.shape();
  auto out_strides = out.strides();
  auto axis_stride = in_strides[axis_];
  size_t axis_size = shape[axis_];
  if (out_strides.size() == in_strides.size()) {
    out_strides.erase(out_strides.begin() + axis_);
  }
  in_strides.erase(in_strides.begin() + axis_);
  shape.erase(shape.begin() + axis_);
  size_t ndim = shape.size();

  // ArgReduce
  constexpr int n_reads = 4;
  auto& encoder = mxcuda::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_CUDA_TYPES(in.dtype(), CTYPE, [&]() {
      if constexpr (!std::is_same_v<CTYPE, cuComplex>) {
        size_t max_threads_per_block = mxcuda::max_threads_per_block(s.device);
        size_t block_dim = std::min(
            mxcuda::ceil_div(axis_size, n_reads), max_threads_per_block);
        // Round up to the closest number divisible by warp size.
        block_dim = mxcuda::ceil_div(block_dim, WARP_SIZE) * WARP_SIZE;
        assert(block_dim <= max_threads_per_block);

        switch (reduce_type_) {
          case ArgReduce::ArgMax:
            mxcuda::arg_reduce_general<CTYPE, mxcuda::ArgMax<CTYPE>>
                <<<out.data_size(), block_dim, 0, stream>>>(
                    in.data<CTYPE>(),
                    out.data<uint32_t>(),
                    shape.data(),
                    in_strides.data(),
                    out_strides.data(),
                    ndim,
                    axis_stride,
                    axis_size);
            break;
          case ArgReduce::ArgMin:
            mxcuda::arg_reduce_general<CTYPE, mxcuda::ArgMin<CTYPE>>
                <<<out.data_size(), block_dim, 0, stream>>>(
                    in.data<CTYPE>(),
                    out.data<uint32_t>(),
                    shape.data(),
                    in_strides.data(),
                    out_strides.data(),
                    ndim,
                    axis_stride,
                    axis_size);
            break;
        }
      } else {
        throw std::runtime_error(fmt::format(
            "Can not arg reduce input with dtype {}",
            dtype_to_string(in.dtype())));
      }
    });
  });
}

void AsStrided::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Broadcast::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void BroadcastAxes::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Full::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto in = inputs[0];
  CopyType ctype;
  if (in.data_size() == 1) {
    ctype = CopyType::Scalar;
  } else if (in.flags().contiguous) {
    ctype = CopyType::Vector;
  } else {
    ctype = CopyType::General;
  }
  copy_gpu(in, out, ctype);
}

NO_GPU(Abs)
NO_GPU(AddMM)
NO_GPU(Arange)
NO_GPU(ArcCos)
NO_GPU(ArcCosh)
NO_GPU(ArcSin)
NO_GPU(ArcSinh)
NO_GPU(ArcTan)
NO_GPU(ArcTanh)
NO_GPU(ArgPartition)
NO_GPU(ArgSort)
NO_GPU(AsType)
NO_GPU(BitwiseInvert)
NO_GPU(BlockMaskedMM)
NO_GPU(Ceil)
NO_GPU_MULTI(Compiled)
NO_GPU(Concatenate)
NO_GPU(Conjugate)
NO_GPU(Contiguous)
NO_GPU(Convolution)
NO_GPU(Copy)
NO_GPU(Cos)
NO_GPU(Cosh)
NO_GPU_MULTI(CustomTransforms)
NO_GPU_MULTI(Depends)
NO_GPU_MULTI(DivMod)
NO_GPU(DynamicSlice)
NO_GPU(DynamicSliceUpdate)
NO_GPU(NumberOfElements)
NO_GPU(Erf)
NO_GPU(ErfInv)
NO_GPU(Exp)
NO_GPU(ExpandDims)
NO_GPU(Expm1)
NO_GPU(FFT)
NO_GPU(Flatten)
NO_GPU(Floor)
NO_GPU(Gather)
NO_GPU(GatherAxis)
NO_GPU(GatherMM)
NO_GPU(GatherQMM)
NO_GPU(Hadamard)
NO_GPU(Imag)
NO_GPU(Load)
NO_GPU(Log)
NO_GPU(Log1p)
NO_GPU(LogicalNot)
NO_GPU_MULTI(LUF)
NO_GPU(Matmul)
NO_GPU(Negative)
NO_GPU(Pad)
NO_GPU(Partition)
NO_GPU_MULTI(QRF)
NO_GPU(QuantizedMatmul)
NO_GPU(RandomBits)
NO_GPU(Real)
NO_GPU(Reduce)
NO_GPU(Reshape)
NO_GPU(Round)
NO_GPU(Scan)
NO_GPU(Scatter)
NO_GPU(ScatterAxis)
NO_GPU(Select)
NO_GPU(Sigmoid)
NO_GPU(Sign)
NO_GPU(Sin)
NO_GPU(Sinh)
NO_GPU(Slice)
NO_GPU(SliceUpdate)
NO_GPU(Softmax)
NO_GPU(Sort)
NO_GPU_MULTI(Split)
NO_GPU(Square)
NO_GPU(Squeeze)
NO_GPU(Sqrt)
NO_GPU(StopGradient)
NO_GPU_MULTI(SVD)
NO_GPU(Tan)
NO_GPU(Tanh)
NO_GPU(Transpose)
NO_GPU(Unflatten)
NO_GPU(Inverse)
NO_GPU(Cholesky)
NO_GPU_MULTI(Eigh)
NO_GPU(View)

namespace fast {
NO_GPU_MULTI(LayerNorm)
NO_GPU_MULTI(LayerNormVJP)
NO_GPU_MULTI(RMSNorm)
NO_GPU_MULTI(RMSNormVJP)
NO_GPU_MULTI(RoPE)
NO_GPU(ScaledDotProductAttention)
NO_GPU_MULTI(AffineQuantize)
NO_GPU_MULTI(CustomKernel)
} // namespace fast

namespace distributed {
NO_GPU_MULTI(AllReduce)
NO_GPU_MULTI(AllGather)
NO_GPU_MULTI(Send)
NO_GPU_MULTI(Recv)
} // namespace distributed

} // namespace mlx::core
