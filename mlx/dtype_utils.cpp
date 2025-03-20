// Copyright © 2025 Apple Inc.

#include "mlx/dtype_utils.h"

namespace mlx::core {

const char* dtype_to_string(Dtype arg) {
#define SPECIALIZE_DtypeToString(cpp_type, dtype) \
  if (dtype == arg) {                             \
    return #dtype;                                \
  }
  MLX_FORALL_DTYPES(SPECIALIZE_DtypeToString)
#undef SPECIALIZE_DtypeToString
  return "(unknown)";
}

} // namespace mlx::core
