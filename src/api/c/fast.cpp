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
#include <af/features.h>
#include <af/image.h>
#include <handle.hpp>
#include <err_common.hpp>
#include <backend.hpp>
#include <fast.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static af_features * fast(af_array const &in, const float thr, const unsigned arc_length, const bool non_max, const float feature_ratio)
{
    return fast<T>(getArray<T>(in), thr, arc_length, non_max, feature_ratio)->get();
}

af_err af_fast(af_features **out, const af_array in, const float thr, const unsigned arc_length, const bool non_max, const float feature_ratio)
{
    try {
        ArrayInfo info = getInfo(in);
        af::dim4 dims  = info.dims();

        ARG_ASSERT(2, (dims[0] >= 7 || dims[1] >= 7));
        ARG_ASSERT(3, thr > 0.0f);
        ARG_ASSERT(4, (arc_length >= 9 && arc_length <= 16));
        ARG_ASSERT(6, (feature_ratio > 0.0f && feature_ratio <= 1.0f));

        dim_type in_ndims = dims.ndims();
        DIM_ASSERT(1, (in_ndims <= 3 && in_ndims >= 2));

        af_features * output;
        af_dtype type  = info.getType();
        switch(type) {
            case f32: output = fast<float >(in, thr, arc_length, non_max, feature_ratio); break;
            case f64: output = fast<double>(in, thr, arc_length, non_max, feature_ratio); break;
            case b8 : output = fast<char  >(in, thr, arc_length, non_max, feature_ratio); break;
            case s32: output = fast<int   >(in, thr, arc_length, non_max, feature_ratio); break;
            case u32: output = fast<uint  >(in, thr, arc_length, non_max, feature_ratio); break;
            case u8 : output = fast<uchar >(in, thr, arc_length, non_max, feature_ratio); break;
            default : TYPE_ERROR(1, type);
        }
        *out = output;
    }
    CATCHALL;

    return AF_SUCCESS;
}
