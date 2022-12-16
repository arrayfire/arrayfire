/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <hsv_rgb.hpp>

#include <kernel/hsv_rgb.hpp>

namespace arrayfire {
namespace opencl {

template<typename T>
Array<T> hsv2rgb(const Array<T>& in) {
    Array<T> out = createEmptyArray<T>(in.dims());
    kernel::hsv2rgb_convert<T>(out, in, true);
    return out;
}

template<typename T>
Array<T> rgb2hsv(const Array<T>& in) {
    Array<T> out = createEmptyArray<T>(in.dims());
    kernel::hsv2rgb_convert<T>(out, in, false);
    return out;
}

#define INSTANTIATE(T)                                \
    template Array<T> hsv2rgb<T>(const Array<T>& in); \
    template Array<T> rgb2hsv<T>(const Array<T>& in);

INSTANTIATE(double)
INSTANTIATE(float)

}  // namespace opencl
}  // namespace arrayfire
