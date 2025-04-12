// Copyright © 2023-2024 Apple Inc.

#include "mlx/backend/metal/metal.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/metal/metal_impl.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

#include <nvtx3/nvtx3.hpp>

namespace mlx::core::metal {

bool is_available() {
  return true;
}

void start_capture(std::string) {}
void stop_capture() {}

void eval(array& arr) {
  nvtx3::scoped_range r("metal::eval");
  auto s = arr.primitive().stream();

  auto outputs = arr.outputs();
  {
    // If the array is a tracer hold a reference
    // to its inputs so they don't get donated
    std::vector<array> inputs;
    if (arr.is_tracer()) {
      inputs = arr.inputs();
    }

    arr.primitive().eval_gpu(arr.inputs(), outputs);
  }
  std::unordered_set<std::shared_ptr<array::Data>> buffers;
  for (auto& in : arr.inputs()) {
    buffers.insert(in.data_shared_ptr());
  }
  for (auto& s : arr.siblings()) {
    buffers.insert(s.data_shared_ptr());
  }
  // Remove the output if it was donated to by an input
  if (auto it = buffers.find(arr.data_shared_ptr()); it != buffers.end()) {
    buffers.erase(it);
  }

  auto& stream = mxcuda::get_stream(s);
  stream.retain_until_completion(std::move(buffers));
  stream.get_encoder().end_eval();
}

void finalize(Stream s) {
  nvtx3::scoped_range r("finalize");
  mxcuda::get_stream(s).finalize();
}

void synchronize(Stream s) {
  nvtx3::scoped_range r("synchronize");
  mxcuda::get_stream(s).synchronize();
}

} // namespace mlx::core::metal
