/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <backend.hpp>
#include <common/err_common.hpp>
#include <features.hpp>
#include <handle.hpp>
#include <sift.hpp>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/features.h>
#include <af/vision.h>

using af::dim4;
using detail::Array;
using detail::createEmptyArray;

template<typename T, typename convAccT>
static void sift(af_features& feat_, af_array& descriptors, const af_array& in,
                 const unsigned n_layers, const float contrast_thr,
                 const float edge_thr, const float init_sigma,
                 const bool double_input, const float img_scale,
                 const float feature_ratio, const bool compute_GLOH) {
    Array<float> x     = createEmptyArray<float>(dim4());
    Array<float> y     = createEmptyArray<float>(dim4());
    Array<float> score = createEmptyArray<float>(dim4());
    Array<float> ori   = createEmptyArray<float>(dim4());
    Array<float> size  = createEmptyArray<float>(dim4());
    Array<float> desc  = createEmptyArray<float>(dim4());

    af_features_t feat;

    feat.n =
        sift<T, convAccT>(x, y, score, ori, size, desc, getArray<T>(in),
                          n_layers, contrast_thr, edge_thr, init_sigma,
                          double_input, img_scale, feature_ratio, compute_GLOH);

    feat.x           = getHandle(x);
    feat.y           = getHandle(y);
    feat.score       = getHandle(score);
    feat.orientation = getHandle(ori);
    feat.size        = getHandle(size);

    feat_       = getFeaturesHandle(feat);
    descriptors = getHandle<float>(desc);
}

af_err af_sift(af_features* feat, af_array* desc, const af_array in,
               const unsigned n_layers, const float contrast_thr,
               const float edge_thr, const float init_sigma,
               const bool double_input, const float img_scale,
               const float feature_ratio) {
    try {
        const ArrayInfo& info = getInfo(in);
        af::dim4 dims         = info.dims();

        ARG_ASSERT(2, (dims[0] >= 15 && dims[1] >= 15 && dims[2] == 1 &&
                       dims[3] == 1));
        ARG_ASSERT(3, n_layers > 0);
        ARG_ASSERT(4, contrast_thr > 0.0f);
        ARG_ASSERT(5, edge_thr >= 1.0f);
        ARG_ASSERT(6, init_sigma > 0.5f);
        ARG_ASSERT(8, img_scale > 0.0f);
        ARG_ASSERT(9, feature_ratio > 0.0f);

        dim_t in_ndims = dims.ndims();
        DIM_ASSERT(1, (in_ndims <= 3 && in_ndims >= 2));

        af_array tmp_desc;
        af_dtype type = info.getType();
        switch (type) {
            case f32:
                sift<float, float>(*feat, tmp_desc, in, n_layers, contrast_thr,
                                   edge_thr, init_sigma, double_input,
                                   img_scale, feature_ratio, false);
                break;
            case f64:
                sift<double, double>(
                    *feat, tmp_desc, in, n_layers, contrast_thr, edge_thr,
                    init_sigma, double_input, img_scale, feature_ratio, false);
                break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*desc, tmp_desc);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_gloh(af_features* feat, af_array* desc, const af_array in,
               const unsigned n_layers, const float contrast_thr,
               const float edge_thr, const float init_sigma,
               const bool double_input, const float img_scale,
               const float feature_ratio) {
    try {
        const ArrayInfo& info = getInfo(in);
        af::dim4 dims         = info.dims();

        ARG_ASSERT(2, (dims[0] >= 15 && dims[1] >= 15 && dims[2] == 1 &&
                       dims[3] == 1));
        ARG_ASSERT(3, n_layers > 0);
        ARG_ASSERT(4, contrast_thr > 0.0f);
        ARG_ASSERT(5, edge_thr >= 1.0f);
        ARG_ASSERT(6, init_sigma > 0.5f);
        ARG_ASSERT(8, img_scale > 0.0f);
        ARG_ASSERT(9, feature_ratio > 0.0f);

        dim_t in_ndims = dims.ndims();
        DIM_ASSERT(1, (in_ndims == 2));

        af_array tmp_desc;
        af_dtype type = info.getType();
        switch (type) {
            case f32:
                sift<float, float>(*feat, tmp_desc, in, n_layers, contrast_thr,
                                   edge_thr, init_sigma, double_input,
                                   img_scale, feature_ratio, true);
                break;
            case f64:
                sift<double, double>(
                    *feat, tmp_desc, in, n_layers, contrast_thr, edge_thr,
                    init_sigma, double_input, img_scale, feature_ratio, true);
                break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*desc, tmp_desc);
    }
    CATCHALL;

    return AF_SUCCESS;
}
