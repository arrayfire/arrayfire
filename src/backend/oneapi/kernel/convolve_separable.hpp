/*******************************************************
 * Copyright (c) 2023, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>

namespace arrayfire {
namespace oneapi {
namespace kernel {

// below shared MAX_*_LEN's are calculated based on
// a maximum shared memory configuration of 48KB per block
// considering complex types as well
constexpr int MAX_SCONV_FILTER_LEN = 31;

template<typename T, typename accT>
void convSep(Param<T> out, const Param<T> sig, const Param<accT> filt,
             const int cDim, const bool expand);

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
