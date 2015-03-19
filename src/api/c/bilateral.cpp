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
#include <bilateral.hpp>
#include <err_common.hpp>

using af::dim4;
using namespace detail;

template<typename inType, typename outType, bool isColor>
static inline af_array bilateral(const af_array &in, const float &sp_sig, const float &chr_sig)
{
    return getHandle(bilateral<inType, outType, isColor>(getArray<inType>(in), sp_sig, chr_sig));
}

template<bool isColor>
static af_err bilateral(af_array *out, const af_array &in, const float &s_sigma, const float &c_sigma)
{
    try {
        ArrayInfo info = getInfo(in);
        af_dtype type  = info.getType();
        af::dim4 dims  = info.dims();

        DIM_ASSERT(1, (dims.ndims()>=2));

        af_array output;
        switch(type) {
            case f64: output = bilateral<double, double, isColor> (in, s_sigma, c_sigma); break;
            case f32: output = bilateral<float ,  float, isColor> (in, s_sigma, c_sigma); break;
            case b8 : output = bilateral<char  ,  float, isColor> (in, s_sigma, c_sigma); break;
            case s32: output = bilateral<int   ,  float, isColor> (in, s_sigma, c_sigma); break;
            case u32: output = bilateral<uint  ,  float, isColor> (in, s_sigma, c_sigma); break;
            case u8 : output = bilateral<uchar ,  float, isColor> (in, s_sigma, c_sigma); break;
            default : TYPE_ERROR(1, type);
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_bilateral(af_array *out, const af_array in, const float spatial_sigma, const float chromatic_sigma, const bool isColor)
{
    if (isColor)
        return bilateral<true>(out,in,spatial_sigma,chromatic_sigma);
    else
        return bilateral<false>(out,in,spatial_sigma,chromatic_sigma);
}
