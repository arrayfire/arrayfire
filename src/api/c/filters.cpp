/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/defines.h>
#include <af/image.h>
#include <af/data.h>
#include <handle.hpp>
#include <err_common.hpp>
#include <backend.hpp>
#include <medfilt.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static af_array medfilt(af_array const &in, dim_t w_len, dim_t w_wid, af_border_type edge_pad)
{
    switch(edge_pad) {
        case AF_PAD_ZERO : return getHandle<T>(medfilt<T, AF_PAD_ZERO>(getArray<T>(in), w_len, w_wid)); break;
        case AF_PAD_SYM  : return getHandle<T>(medfilt<T, AF_PAD_SYM >(getArray<T>(in), w_len, w_wid)); break;
        default          : return getHandle<T>(medfilt<T, AF_PAD_ZERO>(getArray<T>(in), w_len, w_wid)); break;
    }
}

af_err af_medfilt(af_array *out, const af_array in, const dim_t wind_length, const dim_t wind_width, const af_border_type edge_pad)
{
    try {
        ARG_ASSERT(2, (wind_length==wind_width));
        ARG_ASSERT(2, (wind_length>0));
        ARG_ASSERT(3, (wind_width>0));
        ARG_ASSERT(4, (edge_pad>=AF_PAD_ZERO && edge_pad<=AF_PAD_SYM));

        ArrayInfo info = getInfo(in);
        af::dim4 dims  = info.dims();

        dim_t input_ndims = dims.ndims();
        DIM_ASSERT(1, (input_ndims >= 2));

        if (wind_length==1) {
            *out = retain(in);
        } else {
            af_array output;
            af_dtype type  = info.getType();
            switch(type) {
                case f32: output = medfilt<float >(in, wind_length, wind_width, edge_pad); break;
                case f64: output = medfilt<double>(in, wind_length, wind_width, edge_pad); break;
                case b8 : output = medfilt<char  >(in, wind_length, wind_width, edge_pad); break;
                case s32: output = medfilt<int   >(in, wind_length, wind_width, edge_pad); break;
                case u32: output = medfilt<uint  >(in, wind_length, wind_width, edge_pad); break;
                case u8 : output = medfilt<uchar >(in, wind_length, wind_width, edge_pad); break;
                default : TYPE_ERROR(1, type);
            }
            std::swap(*out, output);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_minfilt(af_array *out, const af_array in, const dim_t wind_length,
                  const dim_t wind_width, const af_border_type edge_pad)
{
    try {
        ARG_ASSERT(2, (wind_length==wind_width));
        ARG_ASSERT(2, (wind_length>0));
        ARG_ASSERT(3, (wind_width>0));
        ARG_ASSERT(4, (edge_pad==AF_PAD_ZERO));

        ArrayInfo info = getInfo(in);
        af::dim4 dims  = info.dims();

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
                  const dim_t wind_width, const af_border_type edge_pad)
{
    try {
        ARG_ASSERT(2, (wind_length==wind_width));
        ARG_ASSERT(2, (wind_length>0));
        ARG_ASSERT(3, (wind_width>0));
        ARG_ASSERT(4, (edge_pad==AF_PAD_ZERO));

        ArrayInfo info = getInfo(in);
        af::dim4 dims  = info.dims();

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
