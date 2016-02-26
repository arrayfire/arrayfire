/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <transform.hpp>
#include <math.hpp>
#include <platform.hpp>
#include "transform_interp.hpp"
#include <kernel/transform.hpp>

namespace cpu
{

template<typename T>
Array<T> transform(const Array<T> &in, const Array<float> &transform, const af::dim4 &odims,
                    const af_interp_type method, const bool inverse, const bool perspective)
{
    in.eval();
    transform.eval();

    Array<T> out = createEmptyArray<T>(odims);

    switch(method) {
        case AF_INTERP_NEAREST :
            getQueue().enqueue(kernel::transform<T, AF_INTERP_NEAREST >, out, in, transform,
                    inverse, perspective);
            break;
        case AF_INTERP_BILINEAR:
            getQueue().enqueue(kernel::transform<T, AF_INTERP_BILINEAR>, out, in, transform,
                    inverse, perspective);
            break;
        case AF_INTERP_LOWER   :
            getQueue().enqueue(kernel::transform<T, AF_INTERP_LOWER   >, out, in, transform,
                    inverse, perspective);
            break;
        default: AF_ERROR("Unsupported interpolation type", AF_ERR_ARG); break;
    }

    return out;
}


#define INSTANTIATE(T)                                                              \
template Array<T> transform(const Array<T> &in, const Array<float> &transform,      \
                            const af::dim4 &odims, const af_interp_type method,     \
                            const bool inverse, const bool perspective);


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
