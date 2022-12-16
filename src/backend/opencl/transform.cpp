/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <transform.hpp>

#include <kernel/transform.hpp>

namespace arrayfire {
namespace opencl {

template<typename T>
void transform(Array<T> &out, const Array<T> &in, const Array<float> &tf,
               const af_interp_type method, const bool inverse,
               const bool perspective) {
    switch (method) {
        case AF_INTERP_NEAREST:
        case AF_INTERP_LOWER:
            kernel::transform<T>(out, in, tf, inverse, perspective, method, 1);
            break;
        case AF_INTERP_BILINEAR:
        case AF_INTERP_BILINEAR_COSINE:
            kernel::transform<T>(out, in, tf, inverse, perspective, method, 2);
            break;
        case AF_INTERP_BICUBIC:
        case AF_INTERP_BICUBIC_SPLINE:
            kernel::transform<T>(out, in, tf, inverse, perspective, method, 3);
            break;
        default: AF_ERROR("Unsupported interpolation type", AF_ERR_ARG);
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

}  // namespace opencl
}  // namespace arrayfire
