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
#include <handle.hpp>
#include <backend.hpp>
#include <meanshift.hpp>
#include <err_common.hpp>

using af::dim4;
using namespace detail;

template<typename T, bool is_color>
static inline af_array mean_shift(const af_array &in, const float &s_sigma, const float &c_sigma, const unsigned iter)
{
    return getHandle(meanshift<T, is_color>(getArray<T>(in), s_sigma, c_sigma, iter));
}

template<bool is_color>
af_err mean_shift(af_array *out, const af_array in, const float s_sigma, const float c_sigma, const unsigned iter)
{
    try {
        ARG_ASSERT(2, (s_sigma>=0));
        ARG_ASSERT(3, (c_sigma>=0));
        ARG_ASSERT(4, (iter>0));

        ArrayInfo info = getInfo(in);
        af_dtype type  = info.getType();
        af::dim4 dims  = info.dims();

        DIM_ASSERT(1, (dims.ndims()>=2));
        if (is_color) DIM_ASSERT(1, (dims[2]==3));

        af_array output;
        switch(type) {
            case f32: output = mean_shift<float , is_color>(in, s_sigma, c_sigma, iter); break;
            case f64: output = mean_shift<double, is_color>(in, s_sigma, c_sigma, iter); break;
            case b8 : output = mean_shift<char  , is_color>(in, s_sigma, c_sigma, iter); break;
            case s32: output = mean_shift<int   , is_color>(in, s_sigma, c_sigma, iter); break;
            case u32: output = mean_shift<uint  , is_color>(in, s_sigma, c_sigma, iter); break;
            case u8 : output = mean_shift<uchar , is_color>(in, s_sigma, c_sigma, iter); break;
            default : TYPE_ERROR(1, type);
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_mean_shift(af_array *out, const af_array in, const float spatial_sigma, const float chromatic_sigma, const unsigned iter, const bool is_color)
{
    if (is_color)
        return mean_shift<true >(out, in, spatial_sigma, chromatic_sigma, iter);
    else
        return mean_shift<false>(out, in, spatial_sigma, chromatic_sigma, iter);
}
