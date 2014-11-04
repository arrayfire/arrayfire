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
#include <set.hpp>
#include <copy.hpp>
#include <sort.hpp>
#include <err_opencl.hpp>

namespace opencl
{
    using namespace std;
    using af::dim4;

    template<typename T>
    Array<T>* setUnique(const Array<T> &in,
                        const bool is_sorted)
    {
        OPENCL_NOT_SUPPORTED();
    }

    template<typename T>
    Array<T>* setUnion(const Array<T> &first,
                       const Array<T> &second,
                       const bool is_unique)
    {
        OPENCL_NOT_SUPPORTED();
    }

    template<typename T>
    Array<T>* setIntersect(const Array<T> &first,
                           const Array<T> &second,
                           const bool is_unique)
    {
        OPENCL_NOT_SUPPORTED();
    }

#define INSTANTIATE(T)                                                  \
    template Array<T>* setUnique<T>(const Array<T> &in, const bool is_sorted); \
    template Array<T>* setUnion<T>(const Array<T> &first, const Array<T> &second, const bool is_unique); \
    template Array<T>* setIntersect<T>(const Array<T> &first, const Array<T> &second, const bool is_unique); \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(char)
    INSTANTIATE(uchar)
}
