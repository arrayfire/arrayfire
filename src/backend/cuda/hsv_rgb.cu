/*******************************************************
* Copyright (c) 2014, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <hsv_rgb.hpp>
#include <kernel/hsv_rgb.hpp>
#include <err_cuda.hpp>

using af::dim4;

namespace cuda
{

template<typename T>
Array<T> hsv2rgb(const Array<T>& in)
{
    Array<T> out   = createEmptyArray<T>(in.dims());

    kernel::hsv2rgb_convert<T, true>(out, in);

    return out;
}

template<typename T>
Array<T> rgb2hsv(const Array<T>& in)
{
    Array<T> out   = createEmptyArray<T>(in.dims());

    kernel::hsv2rgb_convert<T, false>(out, in);

    return out;
}

#define INSTANTIATE(T)  \
    template Array<T> hsv2rgb<T>(const Array<T>& in); \
    template Array<T> rgb2hsv<T>(const Array<T>& in); \

INSTANTIATE(double)
INSTANTIATE(float )

}
