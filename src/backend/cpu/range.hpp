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
namespace cpu {
template<typename T>
Array<T> range(const dim4& dim, const int seq_dim = -1);
}  // namespace cpu
}  // namespace arrayfire
