/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
namespace arrayfire {
namespace cuda {
namespace kernel {
// Wrapper functions
template<typename Tk, typename Tv>
void thrustSortByKey(Tk *keyPtr, Tv *valPtr, int elements, bool isAscending);
}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
