/*******************************************************
 * Copyright (c) 2015, ArrayFire
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
#include <susan.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static af_features susan(af_array const &in,
                         const unsigned radius, const float diff_thr, const float geom_thr,
                         const float feature_ratio, const unsigned edge)
{
    Array<float> x = createEmptyArray<float>(dim4());
    Array<float> y = createEmptyArray<float>(dim4());
    Array<float> score = createEmptyArray<float>(dim4());

    af_features_t feat;
    feat.n = susan<T>(x, y, score,
                      getArray<T>(in), radius, diff_thr, geom_thr,
                      feature_ratio, edge);

    Array<float> orientation = createValueArray<float>(feat.n, 0.0);
    Array<float> size = createValueArray<float>(feat.n, 1.0);

    feat.x           = getHandle(x);
    feat.y           = getHandle(y);
    feat.score       = getHandle(score);
    feat.orientation = getHandle(orientation);
    feat.size        = getHandle(size);

    return getFeaturesHandle(feat);
}

af_err af_susan(af_features* out, const af_array in,
                const unsigned radius, const float diff_thr, const float geom_thr,
                const float feature_ratio, const unsigned edge)
{
    try {
        ArrayInfo info = getInfo(in);
        af::dim4 dims  = info.dims();

        ARG_ASSERT(1, dims.ndims()==2);
        ARG_ASSERT(2, radius < 10);
        ARG_ASSERT(2, radius<=edge);
        ARG_ASSERT(3, diff_thr > 0.0f);
        ARG_ASSERT(4, geom_thr > 0.0f);
        ARG_ASSERT(5, (feature_ratio > 0.0f && feature_ratio <= 1.0f));
        ARG_ASSERT(6, (dims[0] >= (dim_t)(2*edge+1) || dims[1] >= (dim_t)(2*edge+1)));

        af_dtype type  = info.getType();
        switch(type) {
            case f32: *out = susan<float >(in, radius, diff_thr, geom_thr, feature_ratio, edge); break;
            case f64: *out = susan<double>(in, radius, diff_thr, geom_thr, feature_ratio, edge); break;
            case b8 : *out = susan<char  >(in, radius, diff_thr, geom_thr, feature_ratio, edge); break;
            case s32: *out = susan<int   >(in, radius, diff_thr, geom_thr, feature_ratio, edge); break;
            case u32: *out = susan<uint  >(in, radius, diff_thr, geom_thr, feature_ratio, edge); break;
            case u8 : *out = susan<uchar >(in, radius, diff_thr, geom_thr, feature_ratio, edge); break;
            default : TYPE_ERROR(1, type);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}
