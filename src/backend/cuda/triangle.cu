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
#include <triangle.hpp>
#include <kernel/triangle.hpp>

using af::dim4;

namespace cuda
{

template<typename T, bool is_upper>
void triangle(Array<T> &out, const Array<T> &in)
{
    kernel::triangle<T, is_upper>(out, in);
}


template<typename T, bool is_upper>
Array<T> triangle(const Array<T> &in)
{
    Array<T> out = createEmptyArray<T>(in.dims());
    triangle<T, is_upper>(out, in);
    return out;
}


#define INSTANTIATE(T)                                                  \
    template void triangle<T, true >(Array<T> &out, const Array<T> &in); \
    template void triangle<T, false>(Array<T> &out, const Array<T> &in); \
    template Array<T> triangle<T, true >(const Array<T> &in);           \
    template Array<T> triangle<T, false>(const Array<T> &in);           \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(intl)
    INSTANTIATE(uintl)
    INSTANTIATE(char)
    INSTANTIATE(uchar)
}
