/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <rotate.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include "transform_interp.hpp"
#include <kernel/rotate.hpp>

namespace cpu
{

template<typename T>
Array<T> rotate(const Array<T> &in, const float theta, const af::dim4 &odims,
                 const af_interp_type method)
{
    in.eval();

    Array<T> out = createEmptyArray<T>(odims);

    switch(method) {
        case AF_INTERP_NEAREST:
            getQueue().enqueue(kernel::rotate<T, AF_INTERP_NEAREST>, out, in, theta);
            break;
        case AF_INTERP_BILINEAR:
            getQueue().enqueue(kernel::rotate<T, AF_INTERP_BILINEAR>, out, in, theta);
            break;
        case AF_INTERP_LOWER:
            getQueue().enqueue(kernel::rotate<T, AF_INTERP_LOWER>, out, in, theta);
            break;
        default:
            AF_ERROR("Unsupported interpolation type", AF_ERR_ARG);
            break;
    }

    return out;
}


#define INSTANTIATE(T)                                                              \
    template Array<T> rotate(const Array<T> &in, const float theta,                 \
                             const af::dim4 &odims, const af_interp_type method);

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
INSTANTIATE(short)
INSTANTIATE(ushort)

}
