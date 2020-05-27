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
#include <medfilt.hpp>
#include <af/data.h>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/image.h>
#include <af/signal.h>

using af::dim4;
using detail::uchar;
using detail::uint;
using detail::ushort;

af_err af_medfilt(af_array *out, const af_array in, const dim_t wind_length,
                  const dim_t wind_width, const af_border_type edge_pad) {
    return af_medfilt2(out, in, wind_length, wind_width, edge_pad);
}

template<typename T>
static af_array medfilt1(af_array const &in, dim_t w_wid,
                         af_border_type edge_pad) {
    return getHandle<T>(
        medfilt1<T>(getArray<T>(in), static_cast<int>(w_wid), edge_pad));
}

af_err af_medfilt1(af_array *out, const af_array in, const dim_t wind_width,
                   const af_border_type edge_pad) {
    try {
        ARG_ASSERT(2, (wind_width > 0));
        ARG_ASSERT(4, (edge_pad >= AF_PAD_ZERO && edge_pad <= AF_PAD_SYM));

        const ArrayInfo &info = getInfo(in);
        af::dim4 dims         = info.dims();

        dim_t input_ndims = dims.ndims();
        DIM_ASSERT(1, (input_ndims >= 1));

        if (wind_width == 1) {
            *out = retain(in);
            return AF_SUCCESS;
        }
        af_array output = nullptr;
        af_dtype type   = info.getType();
        switch (type) {
            case f32: output = medfilt1<float>(in, wind_width, edge_pad); break;
            case f64:
                output = medfilt1<double>(in, wind_width, edge_pad);
                break;
            case b8: output = medfilt1<char>(in, wind_width, edge_pad); break;
            case s32: output = medfilt1<int>(in, wind_width, edge_pad); break;
            case u32: output = medfilt1<uint>(in, wind_width, edge_pad); break;
            case s16: output = medfilt1<short>(in, wind_width, edge_pad); break;
            case u16:
                output = medfilt1<ushort>(in, wind_width, edge_pad);
                break;
            case u8: output = medfilt1<uchar>(in, wind_width, edge_pad); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<typename T>
inline af_array medfilt2(af_array const &in, dim_t w_len, dim_t w_wid,
                         af_border_type edge_pad) {
    return getHandle(medfilt2<T>(getArray<T>(in), static_cast<int>(w_len),
                                 static_cast<int>(w_wid), edge_pad));
}

af_err af_medfilt2(af_array *out, const af_array in, const dim_t wind_length,
                   const dim_t wind_width, const af_border_type edge_pad) {
    try {
        ARG_ASSERT(2, (wind_length == wind_width));
        ARG_ASSERT(2, (wind_length > 0));
        ARG_ASSERT(3, (wind_width > 0));
        ARG_ASSERT(4, (edge_pad >= AF_PAD_ZERO && edge_pad <= AF_PAD_SYM));

        const ArrayInfo &info = getInfo(in);
        af::dim4 dims         = info.dims();

        if (info.isColumn()) {
            return af_medfilt1(out, in, wind_width, edge_pad);
        }

        dim_t input_ndims = dims.ndims();
        DIM_ASSERT(1, (input_ndims >= 2));

        if (wind_length == 1) {
            *out = retain(in);
            return AF_SUCCESS;
        }
        af_array output = nullptr;
        af_dtype type   = info.getType();
        switch (type) {
            case f32:
                output = medfilt2<float>(in, wind_length, wind_width, edge_pad);
                break;
            case f64:
                output =
                    medfilt2<double>(in, wind_length, wind_width, edge_pad);
                break;
            case b8:
                output = medfilt2<char>(in, wind_length, wind_width, edge_pad);
                break;
            case s32:
                output = medfilt2<int>(in, wind_length, wind_width, edge_pad);
                break;
            case u32:
                output = medfilt2<uint>(in, wind_length, wind_width, edge_pad);
                break;
            case s16:
                output = medfilt2<short>(in, wind_length, wind_width, edge_pad);
                break;
            case u16:
                output =
                    medfilt2<ushort>(in, wind_length, wind_width, edge_pad);
                break;
            case u8:
                output = medfilt2<uchar>(in, wind_length, wind_width, edge_pad);
                break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_minfilt(af_array *out, const af_array in, const dim_t wind_length,
                  const dim_t wind_width, const af_border_type edge_pad) {
    try {
        ARG_ASSERT(2, (wind_length == wind_width));
        ARG_ASSERT(2, (wind_length > 0));
        ARG_ASSERT(3, (wind_width > 0));
        ARG_ASSERT(4, (edge_pad == AF_PAD_ZERO));

        const ArrayInfo &info = getInfo(in);
        af::dim4 dims         = info.dims();

        dim_t input_ndims = dims.ndims();
        DIM_ASSERT(1, (input_ndims >= 2));

        af_array mask;
        dim_t wdims[] = {wind_length, wind_width};
        AF_CHECK(af_constant(&mask, 1, 2, wdims, info.getType()));

        AF_CHECK(af_erode(out, in, mask));

        AF_CHECK(af_release_array(mask));
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_maxfilt(af_array *out, const af_array in, const dim_t wind_length,
                  const dim_t wind_width, const af_border_type edge_pad) {
    try {
        ARG_ASSERT(2, (wind_length == wind_width));
        ARG_ASSERT(2, (wind_length > 0));
        ARG_ASSERT(3, (wind_width > 0));
        ARG_ASSERT(4, (edge_pad == AF_PAD_ZERO));

        const ArrayInfo &info = getInfo(in);
        af::dim4 dims         = info.dims();

        dim_t input_ndims = dims.ndims();
        DIM_ASSERT(1, (input_ndims >= 2));

        af_array mask;
        dim_t wdims[] = {wind_length, wind_width};
        AF_CHECK(af_constant(&mask, 1, 2, wdims, info.getType()));

        AF_CHECK(af_dilate(out, in, mask));

        AF_CHECK(af_release_array(mask));
    }
    CATCHALL;

    return AF_SUCCESS;
}
