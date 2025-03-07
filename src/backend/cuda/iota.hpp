/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <Array.hpp>

namespace arrayfire {
namespace cuda {
template<typename T>
Array<T> iota(const dim4 &dim, const dim4 &tile_dims = dim4(1));
}  // namespace cuda
}  // namespace arrayfire
