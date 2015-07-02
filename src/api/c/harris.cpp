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
#include <harris.hpp>

using af::dim4;
using namespace detail;

template<typename T, typename convAccT>
static af_features harris(af_array const &in, const unsigned max_corners,
                          const float min_response, const float sigma,
                          const unsigned filter_len, const float k_thr)
{
    Array<float> x = createEmptyArray<float>(dim4());
    Array<float> y = createEmptyArray<float>(dim4());
    Array<float> score = createEmptyArray<float>(dim4());

    af_features_t feat;
    feat.n = harris<T, convAccT>(x, y, score,
                                 getArray<T>(in), max_corners, min_response,
                                 sigma, filter_len, k_thr);

    Array<float> orientation = createValueArray<float>(feat.n, 0.0);
    Array<float> size = createValueArray<float>(feat.n, 1.0);

    feat.x           = getHandle(x);
    feat.y           = getHandle(y);
    feat.score       = getHandle(score);
    feat.orientation = getHandle(orientation);
    feat.size        = getHandle(size);

    return getFeaturesHandle(feat);
}


af_err af_harris(af_features *out, const af_array in, const unsigned max_corners,
                 const float min_response, const float sigma,
                 const unsigned block_size, const float k_thr)
{
    try {
        ArrayInfo info = getInfo(in);
        af::dim4 dims  = info.dims();
        dim_t in_ndims = dims.ndims();

        unsigned filter_len = (block_size == 0) ? floor(6.f * sigma) : block_size;
        if (block_size == 0 && filter_len % 2 == 0)
            filter_len--;

        const unsigned edge = (block_size > 0) ? block_size / 2 : filter_len / 2;

        DIM_ASSERT(1, (in_ndims == 2));
        ARG_ASSERT(1, (dims[0] >= (dim_t)(2*edge+1) || dims[1] >= (dim_t)(2*edge+1)));
        ARG_ASSERT(3, (max_corners > 0) || (min_response > 0.0f));
        ARG_ASSERT(7, (k_thr >= 0.01f));
        // Upper limits for sigma and block_size are due to convolve2 template
        // at maximum length of 31 elements for the filter in OpenCL
        ARG_ASSERT(4, (block_size > 2) || (sigma >= 0.5f && sigma <= 5.f));
        ARG_ASSERT(5, (block_size <= 32));

        af_dtype type  = info.getType();
        switch(type) {
            case f64: *out = harris<double, double>(in, max_corners, min_response, sigma, filter_len, k_thr); break;
            case f32: *out = harris<float , float >(in, max_corners, min_response, sigma, filter_len, k_thr); break;
            default : TYPE_ERROR(1, type);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}
