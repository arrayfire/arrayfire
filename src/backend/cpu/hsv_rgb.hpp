/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

namespace arrayfire {
namespace cpu {

template<typename T>
Array<T> hsv2rgb(const Array<T>& in);

template<typename T>
Array<T> rgb2hsv(const Array<T>& in);

}  // namespace cpu
}  // namespace arrayfire
