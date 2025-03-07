/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

namespace arrayfire {
namespace opencl {
template<typename T>
Array<float> moments(const Array<T> &in, const af_moment_type moment);
}  // namespace opencl
}  // namespace arrayfire
