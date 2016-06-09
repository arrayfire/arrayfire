/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include <interopManager.hpp>
#include <kernel/moments.hpp>

using af::dim4;

namespace cuda
{

template<typename T>
Array<float> moments(const Array<T> &in, const af_moment_type moment)
{
    dim4 odims, idims = in.dims();
    odims[0] = odims[1] = odims[2] = odims[3] = 1;
    if(idims[2] != 1) {
        odims[0] = idims[2];
    }
    if(idims[3] != 1) {
        odims[1] = idims[3];
    }

    in.eval();
    Array<float> out = createValueArray<float>(odims, 0.f);

    switch(moment) {
        case AF_MOMENT_M00:
            kernel::moments<T, AF_MOMENT_M00>(out, in);
            break;
        case AF_MOMENT_M01:
            kernel::moments<T, AF_MOMENT_M01>(out, in);
            break;
        case AF_MOMENT_M10:
            kernel::moments<T, AF_MOMENT_M10>(out, in);
            break;
        case AF_MOMENT_M11:
            kernel::moments<T, AF_MOMENT_M11>(out, in);
            break;
        default:  break;
    }

    return out;
}

#define INSTANTIATE(T)                                                          \
    template Array<float>  moments<T>(const Array<T> &in, const af_moment_type moment);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(ushort)
INSTANTIATE(short)

}
