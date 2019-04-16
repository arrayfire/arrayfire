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
#include <transform.hpp>
#include <stdexcept>

namespace cuda {
template<typename T>
Array<T> transform(const Array<T> &in, const Array<float> &tf,
                   const af::dim4 &odims, const af_interp_type method,
                   const bool inverse, const bool perspective) {
    Array<T> out = createEmptyArray<T>(odims);

    switch (method) {
        case AF_INTERP_NEAREST:
        case AF_INTERP_LOWER:
            kernel::transform<T, 1>(out, in, tf, inverse, perspective, method);
            break;
        case AF_INTERP_BILINEAR:
        case AF_INTERP_BILINEAR_COSINE:
            kernel::transform<T, 2>(out, in, tf, inverse, perspective, method);
            break;
        case AF_INTERP_BICUBIC:
        case AF_INTERP_BICUBIC_SPLINE:
            kernel::transform<T, 3>(out, in, tf, inverse, perspective, method);
            break;
        default: AF_ERROR("Unsupported interpolation type", AF_ERR_ARG);
    }

    return out;
}

#define INSTANTIATE(T)                                                      \
    template Array<T> transform(const Array<T> &in, const Array<float> &tf, \
                                const af::dim4 &odims,                      \
                                const af_interp_type method,                \
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
}  // namespace cuda
