/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <rotate.hpp>
#include <af/image.h>
#include <cmath>

using af::dim4;
using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;
using std::cos;
using std::fabs;
using std::sin;

template<typename T>
static inline af_array rotate(const af_array in, const float theta,
                              const af::dim4 &odims,
                              const af_interp_type method) {
    return getHandle(rotate<T>(castArray<T>(in), theta, odims, method));
}

af_err af_rotate(af_array *out, const af_array in, const float theta,
                 const bool crop, const af_interp_type method) {
    try {
        dim_t odims0 = 0, odims1 = 0;

        const ArrayInfo &info = getInfo(in);
        af::dim4 idims        = info.dims();

        if (!crop) {
            odims0 = idims[0] * fabs(cos(theta)) + idims[1] * fabs(sin(theta));
            odims1 = idims[1] * fabs(cos(theta)) + idims[0] * fabs(sin(theta));
        } else {
            odims0 = idims[0];
            odims1 = idims[1];
        }

        af_dtype itype = info.getType();

        ARG_ASSERT(4, method == AF_INTERP_NEAREST ||
                          method == AF_INTERP_BILINEAR ||
                          method == AF_INTERP_BILINEAR_COSINE ||
                          method == AF_INTERP_BICUBIC ||
                          method == AF_INTERP_BICUBIC_SPLINE ||
                          method == AF_INTERP_LOWER);

        if (idims.elements() == 0) { return af_retain_array(out, in); }
        DIM_ASSERT(1, idims.elements() > 0);

        af::dim4 odims(odims0, odims1, idims[2], idims[3]);

        af_array output = 0;
        switch (itype) {
            case f32: output = rotate<float>(in, theta, odims, method); break;
            case f64: output = rotate<double>(in, theta, odims, method); break;
            case c32: output = rotate<cfloat>(in, theta, odims, method); break;
            case c64: output = rotate<cdouble>(in, theta, odims, method); break;
            case s32: output = rotate<int>(in, theta, odims, method); break;
            case u32: output = rotate<uint>(in, theta, odims, method); break;
            case s64: output = rotate<intl>(in, theta, odims, method); break;
            case u64: output = rotate<uintl>(in, theta, odims, method); break;
            case s16: output = rotate<short>(in, theta, odims, method); break;
            case u16: output = rotate<ushort>(in, theta, odims, method); break;
            case u8:
            case b8: output = rotate<uchar>(in, theta, odims, method); break;
            default: TYPE_ERROR(1, itype);
        }
        std::swap(*out, output);
    }
    CATCHALL

    return AF_SUCCESS;
}
