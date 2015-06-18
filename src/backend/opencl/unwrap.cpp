/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <unwrap.hpp>
#include <kernel/unwrap.hpp>
#include <stdexcept>
#include <err_opencl.hpp>

namespace opencl
{
    template<typename T>
    Array<T> unwrap(const Array<T> &in, const dim_t wx, const dim_t wy,
                    const dim_t sx, const dim_t sy)
    {
        af::dim4 idims = in.dims();

        dim_t nx = divup(idims[0] - wx, sx) + (sx >= idims[0] ? 0 : 1);
        dim_t ny = divup(idims[1] - wy, sy) + (sy >= idims[1] ? 0 : 1);

        af::dim4 odims(wx * wy, nx * ny, idims[2], idims[3]);

        // Create output placeholder
        Array<T> outArray = createEmptyArray<T>(odims);

        if(odims[0] <= 16) {
            kernel::unwrap<T, 16 >(outArray, in, wx, wy, sx, sy);
        } else if (odims[0] <= 32) {
            kernel::unwrap<T, 32 >(outArray, in, wx, wy, sx, sy);
        } else if (odims[0] <= 64) {
            kernel::unwrap<T, 64 >(outArray, in, wx, wy, sx, sy);
        } else if(odims[0] <= 128) {
            kernel::unwrap<T, 128>(outArray, in, wx, wy, sx, sy);
        } else {
            kernel::unwrap<T, 256>(outArray, in, wx, wy, sx, sy);
        }

        return outArray;
    }


#define INSTANTIATE(T)                                                                  \
    template Array<T> unwrap<T> (const Array<T> &in, const dim_t wx, const dim_t wy,    \
                                 const dim_t sx, const dim_t sy);


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

