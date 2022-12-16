/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <kernel/scan_dim_by_key_impl.hpp>
#include <kernel/scan_first_by_key_impl.hpp>

// This file instantiates scan_dim_by_key as separate object files from CMake
// The line below is read by CMake to determenine the instantiations
// SBK_BINARY_OPS:af_add_t af_mul_t af_max_t af_min_t

namespace arrayfire {
namespace cuda {
namespace kernel {
// clang-format off
INSTANTIATE_SCAN_FIRST_BY_KEY_OP( @SBK_BINARY_OP@ )
INSTANTIATE_SCAN_DIM_BY_KEY_OP( @SBK_BINARY_OP@ )
// clang-format on
}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
