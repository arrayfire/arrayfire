/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <kernel/transform.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <transform.hpp>

namespace arrayfire {
namespace cpu {

template<typename T>
void transform(Array<T> &out, const Array<T> &in, const Array<float> &tf,
               const af_interp_type method, const bool inverse,
               const bool perspective) {
    out.eval();
    in.eval();
    tf.eval();

    switch (method) {
        case AF_INTERP_NEAREST:
        case AF_INTERP_LOWER:
            getQueue().enqueue(kernel::transform<T, 1>, out, in, tf, inverse,
                               perspective, method);
            break;
        case AF_INTERP_BILINEAR:
        case AF_INTERP_BILINEAR_COSINE:
            getQueue().enqueue(kernel::transform<T, 2>, out, in, tf, inverse,
                               perspective, method);
            break;
        case AF_INTERP_BICUBIC:
        case AF_INTERP_BICUBIC_SPLINE:
            getQueue().enqueue(kernel::transform<T, 3>, out, in, tf, inverse,
                               perspective, method);
            break;
        default: AF_ERROR("Unsupported interpolation type", AF_ERR_ARG); break;
    }
}

#define INSTANTIATE(T)                                                       \
    template void transform(Array<T> &out, const Array<T> &in,               \
                            const Array<float> &tf,                          \
                            const af_interp_type method, const bool inverse, \
                            const bool perspective);

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
