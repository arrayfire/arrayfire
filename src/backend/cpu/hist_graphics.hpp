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
#include <common/graphics_common.hpp>

namespace arrayfire {
namespace cpu {

template<typename T>
void copy_histogram(const Array<T> &data, fg_histogram hist);

}  // namespace cpu
}  // namespace arrayfire
