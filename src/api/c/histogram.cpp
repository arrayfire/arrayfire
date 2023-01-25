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
#include <histogram.hpp>
#include <af/dim4.hpp>
#include <af/image.h>

using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename T>
inline af_array histogram(const af_array in, const unsigned &nbins,
                          const double &minval, const double &maxval,
                          const bool islinear) {
    return getHandle(
        histogram<T>(getArray<T>(in), nbins, minval, maxval, islinear));
}

af_err af_histogram(af_array *out, const af_array in, const unsigned nbins,
                    const double minval, const double maxval) {
    try {
        const ArrayInfo &info = getInfo(in);
        af_dtype type         = info.getType();

        if (info.ndims() == 0) { return af_retain_array(out, in); }

        af_array output;
        switch (type) {
            case f32:
                output = histogram<float>(in, nbins, minval, maxval,
                                          info.isLinear());
                break;
            case f64:
                output = histogram<double>(in, nbins, minval, maxval,
                                           info.isLinear());
                break;
            case b8:
                output =
                    histogram<char>(in, nbins, minval, maxval, info.isLinear());
                break;
            case s32:
                output =
                    histogram<int>(in, nbins, minval, maxval, info.isLinear());
                break;
            case u32:
                output =
                    histogram<uint>(in, nbins, minval, maxval, info.isLinear());
                break;
            case s16:
                output = histogram<short>(in, nbins, minval, maxval,
                                          info.isLinear());
                break;
            case u16:
                output = histogram<ushort>(in, nbins, minval, maxval,
                                           info.isLinear());
                break;
            case s64:
                output =
                    histogram<intl>(in, nbins, minval, maxval, info.isLinear());
                break;
            case u64:
                output = histogram<uintl>(in, nbins, minval, maxval,
                                          info.isLinear());
                break;
            case u8:
                output = histogram<uchar>(in, nbins, minval, maxval,
                                          info.isLinear());
                break;
            case f16:
                output = histogram<arrayfire::common::half>(
                    in, nbins, minval, maxval, info.isLinear());
                break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
