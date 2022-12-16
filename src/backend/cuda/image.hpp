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
namespace cuda {

template<typename T>
void copy_image(const Array<T> &in, fg_image image);

}  // namespace cuda
}  // namespace arrayfire
