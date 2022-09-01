/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <hsv_rgb.hpp>

#include <err_oneapi.hpp>

namespace oneapi {

template<typename T>
Array<T> hsv2rgb(const Array<T>& in) {
    ONEAPI_NOT_SUPPORTED("");
    Array<T> out = createEmptyArray<T>(in.dims());
    return out;
}

template<typename T>
Array<T> rgb2hsv(const Array<T>& in) {
    ONEAPI_NOT_SUPPORTED("");
    Array<T> out = createEmptyArray<T>(in.dims());
    return out;
}

#define INSTANTIATE(T)                                \
    template Array<T> hsv2rgb<T>(const Array<T>& in); \
    template Array<T> rgb2hsv<T>(const Array<T>& in);

INSTANTIATE(double)
INSTANTIATE(float)

}  // namespace oneapi
