/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <backend.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <hsv_rgb.hpp>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/image.h>

using af::dim4;
using detail::Array;
using detail::hsv2rgb;
using detail::rgb2hsv;

template<typename T, bool isHSV2RGB>
static af_array convert(const af_array& in) {
    const Array<T> input = getArray<T>(in);
    if (isHSV2RGB) {
        return getHandle<T>(hsv2rgb<T>(input));
    } else {
        return getHandle<T>(rgb2hsv<T>(input));
    }
}

template<bool isHSV2RGB>
af_err convert(af_array* out, const af_array& in) {
    try {
        const ArrayInfo& info = getInfo(in);
        af_dtype iType        = info.getType();
        af::dim4 inputDims    = info.dims();

        if (info.ndims() == 0) {
            return af_create_handle(out, 0, nullptr, iType);
        }

        ARG_ASSERT(1, (inputDims.ndims() >= 3));

        af_array output = 0;
        switch (iType) {
            case f64: output = convert<double, isHSV2RGB>(in); break;
            case f32: output = convert<float, isHSV2RGB>(in); break;
            default: TYPE_ERROR(1, iType); break;
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_hsv2rgb(af_array* out, const af_array in) {
    return convert<true>(out, in);
}

af_err af_rgb2hsv(af_array* out, const af_array in) {
    return convert<false>(out, in);
}
