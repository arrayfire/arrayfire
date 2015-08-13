/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/dim4.hpp>
#include <Array.hpp>
#include <diff.hpp>
#include <kernel/diff.hpp>
#include <stdexcept>

namespace opencl
{
    template<typename T, bool isDiff2>
    static Array<T> diff(const Array<T> &in, const int dim)
    {
        const af::dim4 iDims = in.dims();
        af::dim4 oDims = iDims;
        oDims[dim] -= (isDiff2 + 1);

        if(iDims.elements() == 0 || oDims.elements() == 0) {
            throw std::runtime_error("Elements are 0");
        }

        Array<T> out = createEmptyArray<T>(oDims);

        switch (dim) {

            case (0):    kernel::diff<T, 0, isDiff2>(out, in, in.ndims());
                         break;

            case (1):    kernel::diff<T, 1, isDiff2>(out, in, in.ndims());
                         break;

            case (2):    kernel::diff<T, 2, isDiff2>(out, in, in.ndims());
                         break;

            case (3):    kernel::diff<T, 3, isDiff2>(out, in, in.ndims());
                         break;
        }

        return out;
    }

    template<typename T>
    Array<T> diff1(const Array<T> &in, const int dim)
    {
        return diff<T, false>(in, dim);
    }

    template<typename T>
    Array<T> diff2(const Array<T> &in, const int dim)
    {
        return diff<T, true>(in, dim);
    }

#define INSTANTIATE(T)                                                 \
    template Array<T> diff1<T>  (const Array<T> &in, const int dim);   \
    template Array<T> diff2<T>  (const Array<T> &in, const int dim);   \


    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
    INSTANTIATE(intl)
    INSTANTIATE(uintl)
    INSTANTIATE(char)
}
