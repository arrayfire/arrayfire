/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <kernel/thrust_sort_by_key_impl.hpp>

// This file instantiates sort_by_key as separate object files from CMake
// The 3 lines below are read by CMake to determenine the instantiations
// SBK_TYPES:float double int uint intl uintl short ushort char uchar
// SBK_INSTS:0 1

namespace arrayfire {
namespace cuda {
namespace kernel {
// clang-format off
@INSTANTIATESBK_INST@ ( @SBK_TYPE@ )
// clang-format on
}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
