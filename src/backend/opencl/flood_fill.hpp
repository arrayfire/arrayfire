/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Array.hpp>
#include <af/defines.h>

namespace arrayfire {
namespace opencl {
template<typename T>
Array<T> floodFill(const Array<T>& image, const Array<uint>& seedsX,
                   const Array<uint>& seedsY, const T newValue,
                   const T lowValue, const T highValue,
                   const af::connectivity nlookup = AF_CONNECTIVITY_8);
}  // namespace opencl
}  // namespace arrayfire
