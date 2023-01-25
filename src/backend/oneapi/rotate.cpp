/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <err_oneapi.hpp>
#include <rotate.hpp>

#include <kernel/rotate.hpp>

namespace arrayfire {
namespace oneapi {
template<typename T>
Array<T> rotate(const Array<T> &in, const float theta, const af::dim4 &odims,
                const af_interp_type method) {
    Array<T> out = createEmptyArray<T>(odims);

    switch (method) {
        case AF_INTERP_NEAREST:
        case AF_INTERP_LOWER:
            if constexpr (!(std::is_same_v<T, double> ||
                            std::is_same_v<T, cdouble>)) {
                kernel::rotate<T>(out, in, theta, method, 1);
            }
            break;
        case AF_INTERP_BILINEAR:
        case AF_INTERP_BILINEAR_COSINE:
            if constexpr (!(std::is_same_v<T, double> ||
                            std::is_same_v<T, cdouble>)) {
                kernel::rotate<T>(out, in, theta, method, 2);
            }
            break;
        case AF_INTERP_BICUBIC:
        case AF_INTERP_BICUBIC_SPLINE:
            if constexpr (!(std::is_same_v<T, double> ||
                            std::is_same_v<T, cdouble>)) {
                kernel::rotate<T>(out, in, theta, method, 3);
            }
            break;
        default: AF_ERROR("Unsupported interpolation type", AF_ERR_ARG);
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
}  // namespace oneapi
}  // namespace arrayfire
