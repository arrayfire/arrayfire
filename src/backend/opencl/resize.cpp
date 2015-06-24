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
#include <resize.hpp>
#include <kernel/resize.hpp>
#include <stdexcept>

namespace opencl
{
    template<typename T>
    Array<T> resize(const Array<T> &in, const dim_t odim0, const dim_t odim1,
                    const af_interp_type method)
    {
        const af::dim4 iDims = in.dims();
        af::dim4 oDims(odim0, odim1, iDims[2], iDims[3]);

        Array<T> out = createEmptyArray<T>(oDims);

        switch(method) {
            case AF_INTERP_NEAREST:
                kernel::resize<T, AF_INTERP_NEAREST> (out, in);
                break;
            case AF_INTERP_BILINEAR:
                kernel::resize<T, AF_INTERP_BILINEAR>(out, in);
                break;
            case AF_INTERP_LOWER:
                kernel::resize<T, AF_INTERP_LOWER>(out, in);
                break;
            default:
                break;
        }
        return out;
    }


#define INSTANTIATE(T)                                                  \
    template Array<T> resize<T> (const Array<T> &in,                    \
                                 const dim_t odim0, const dim_t odim1, \
                                 const af_interp_type method);


    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(intl)
    INSTANTIATE(uintl)
    INSTANTIATE(uchar)
    INSTANTIATE(char)
}
