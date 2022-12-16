/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/err_common.hpp>
#include <handle.hpp>
#include <imgproc_common.hpp>
#include <af/defines.h>
#include <af/image.h>

using af::dim4;
using arrayfire::common::integralImage;
using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename To, typename Ti>
inline af_array sat(const af_array& in) {
    return getHandle<To>(integralImage<To, Ti>(getArray<Ti>(in)));
}

af_err af_sat(af_array* out, const af_array in) {
    try {
        const ArrayInfo& info = getInfo(in);
        const dim4& dims      = info.dims();

        ARG_ASSERT(1, (dims.ndims() >= 2));

        af_dtype inputType = info.getType();

        af_array output = 0;
        switch (inputType) {
            case f64: output = sat<double, double>(in); break;
            case f32: output = sat<float, float>(in); break;
            case s32: output = sat<int, int>(in); break;
            case u32: output = sat<uint, uint>(in); break;
            case b8: output = sat<int, char>(in); break;
            case u8: output = sat<uint, uchar>(in); break;
            case s64: output = sat<intl, intl>(in); break;
            case u64: output = sat<uintl, uintl>(in); break;
            case s16: output = sat<int, short>(in); break;
            case u16: output = sat<uint, ushort>(in); break;
            default: TYPE_ERROR(1, inputType);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
