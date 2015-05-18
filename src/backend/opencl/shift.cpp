/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <shift.hpp>
#include <kernel/shift.hpp>
#include <stdexcept>
#include <err_opencl.hpp>

namespace opencl
{
    template<typename T>
    Array<T> shift(const Array<T> &in, const int sdims[4])
    {
        if ((std::is_same<T, double>::value || std::is_same<T, cdouble>::value) &&
            !isDoubleSupported(getActiveDeviceId())) {
            OPENCL_NOT_SUPPORTED();
        }
        const af::dim4 iDims = in.dims();
        af::dim4 oDims = iDims;

        Array<T> out = createEmptyArray<T>(oDims);

        kernel::shift<T>(out, in, sdims);

        return out;
    }

#define INSTANTIATE(T)                                                  \
    template Array<T> shift<T>(const Array<T> &in, const int sdims[4]);     \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
    INSTANTIATE(char)
}
