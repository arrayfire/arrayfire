/*******************************************************
* Copyright (c) 2014, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#include <af/dim4.hpp>
#include <Array.hpp>
#include <ArrayInfo.hpp>
#include <hsv_rgb.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <kernel/hsv_rgb.hpp>

using af::dim4;

namespace cpu
{

template<typename T>
Array<T> hsv2rgb(const Array<T>& in)
{
    in.eval();

    Array<T> out = createEmptyArray<T>(in.dims());

    getQueue().enqueue(kernel::hsv2rgb<T>, out, in);

    return out;
}

template<typename T>
Array<T> rgb2hsv(const Array<T>& in)
{
    in.eval();

    Array<T> out = createEmptyArray<T>(in.dims());

    getQueue().enqueue(kernel::rgb2hsv<T>, out, in);

    return out;
}

#define INSTANTIATE(T)  \
    template Array<T> hsv2rgb<T>(const Array<T>& in); \
    template Array<T> rgb2hsv<T>(const Array<T>& in); \

INSTANTIATE(double)
INSTANTIATE(float )

}
