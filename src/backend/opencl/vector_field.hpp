/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/graphics_common.hpp>

namespace arrayfire {
namespace opencl {

template<typename T>
void copy_vector_field(const Array<T> &points, const Array<T> &directions,
                       fg_vector_field vfield);
}  // namespace opencl
}  // namespace arrayfire
