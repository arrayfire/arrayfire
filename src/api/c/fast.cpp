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
#include <af/vision.h>
#include <handle.hpp>
#include <err_common.hpp>
#include <backend.hpp>
#include <features.hpp>
#include <fast.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static af_features fast(af_array const &in, const float thr,
                        const unsigned arc_length, const bool non_max,
                        const float feature_ratio, const unsigned edge)
{
    Array<float> x = createEmptyArray<float>(dim4());
    Array<float> y = createEmptyArray<float>(dim4());
    Array<float> score = createEmptyArray<float>(dim4());

    af_features_t feat;
    feat.n = fast<T>(x, y, score,
                     getArray<T>(in), thr,
                     arc_length, non_max, feature_ratio, edge);

    Array<float> orientation = createValueArray<float>(feat.n, 0.0);
    Array<float> size = createValueArray<float>(feat.n, 1.0);

    feat.x           = getHandle(x);
    feat.y           = getHandle(y);
    feat.score       = getHandle(score);
    feat.orientation = getHandle(orientation);
    feat.size        = getHandle(size);

    return getFeaturesHandle(feat);
}


af_err af_fast(af_features *out, const af_array in, const float thr,
               const unsigned arc_length, const bool non_max,
               const float feature_ratio, const unsigned edge)
{
    try {
        ArrayInfo info = getInfo(in);
        af::dim4 dims  = info.dims();

        ARG_ASSERT(2, (dims[0] >= (dim_t)(2*edge+1) || dims[1] >= (dim_t)(2*edge+1)));
        ARG_ASSERT(3, thr > 0.0f);
        ARG_ASSERT(4, (arc_length >= 9 && arc_length <= 16));
        ARG_ASSERT(6, (feature_ratio > 0.0f && feature_ratio <= 1.0f));

        dim_t in_ndims = dims.ndims();
        DIM_ASSERT(1, (in_ndims <= 3 && in_ndims >= 2));

        af_dtype type  = info.getType();
        switch(type) {
            case f32: *out = fast<float >(in, thr, arc_length, non_max, feature_ratio, edge); break;
            case f64: *out = fast<double>(in, thr, arc_length, non_max, feature_ratio, edge); break;
            case b8 : *out = fast<char  >(in, thr, arc_length, non_max, feature_ratio, edge); break;
            case s32: *out = fast<int   >(in, thr, arc_length, non_max, feature_ratio, edge); break;
            case u32: *out = fast<uint  >(in, thr, arc_length, non_max, feature_ratio, edge); break;
            case u8 : *out = fast<uchar >(in, thr, arc_length, non_max, feature_ratio, edge); break;
            default : TYPE_ERROR(1, type);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}
