/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <Array.hpp>
#include <err_cpu.hpp>
#include <platform.hpp>
#include <kernel/moments.hpp>
#include <queue.hpp>

using af::dim4;

namespace cpu
{

template<typename T>
Array<float> moments(const Array<T> &in, const af_moment_type moment)
{
    dim4 odims, idims = in.dims();
    odims[0] = idims[2];
    odims[1] = idims[3];
    odims[2] = odims[3] = 1;

    in.eval();
    Array<float> out = createEmptyArray<float>(odims);

    switch(moment) {
        case AF_MOMENT_M00:
            getQueue().enqueue(kernel::moments<T, AF_MOMENT_M00>, out, in);
            break;
        case AF_MOMENT_M01:
            getQueue().enqueue(kernel::moments<T, AF_MOMENT_M01>, out, in);
            break;
        case AF_MOMENT_M10:
            getQueue().enqueue(kernel::moments<T, AF_MOMENT_M10>, out, in);
            break;
        case AF_MOMENT_M11:
            getQueue().enqueue(kernel::moments<T, AF_MOMENT_M11>, out, in);
            break;
        default:  break;
    }
    getQueue().sync();
    return out;
}


#define INSTANTIATE(T)  \
    template Array<float> moments<T>(const Array<T> &in, const af_moment_type moment);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(ushort)
INSTANTIATE(short)

}

