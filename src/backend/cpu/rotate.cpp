/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <kernel/rotate.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <rotate.hpp>

namespace arrayfire {
namespace cpu {

template<typename T>
Array<T> rotate(const Array<T> &in, const float theta, const af::dim4 &odims,
                const af_interp_type method) {
    Array<T> out = createEmptyArray<T>(odims);

    switch (method) {
        case AF_INTERP_NEAREST:
        case AF_INTERP_LOWER:
            getQueue().enqueue(kernel::rotate<T, 1>, out, in, theta, method);
            break;
        case AF_INTERP_BILINEAR:
        case AF_INTERP_BILINEAR_COSINE:
            getQueue().enqueue(kernel::rotate<T, 2>, out, in, theta, method);
            break;
        case AF_INTERP_BICUBIC:
        case AF_INTERP_BICUBIC_SPLINE:
            getQueue().enqueue(kernel::rotate<T, 3>, out, in, theta, method);
            break;
        default: AF_ERROR("Unsupported interpolation type", AF_ERR_ARG); break;
    }

    return out;
}

#define INSTANTIATE(T)                                              \
    template Array<T> rotate(const Array<T> &in, const float theta, \
                             const af::dim4 &odims,                 \
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
INSTANTIATE(short)
INSTANTIATE(ushort)

}  // namespace cpu
}  // namespace arrayfire
