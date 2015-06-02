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
#include <orb.hpp>
#include <features.hpp>

using af::dim4;
using namespace detail;

template<typename T, typename convAccT>
static void orb(af_features& feat_, af_array& descriptor,
                const af_array& in, const float fast_thr,
                const unsigned max_feat, const float scl_fctr,
                const unsigned levels, const bool blur_img)
{
    Array<float> x     = createEmptyArray<float>(dim4());
    Array<float> y     = createEmptyArray<float>(dim4());
    Array<float> score = createEmptyArray<float>(dim4());
    Array<float> ori   = createEmptyArray<float>(dim4());
    Array<float> size  = createEmptyArray<float>(dim4());
    Array<uint > desc  = createEmptyArray<uint >(dim4());

    af_features_t feat;

    feat.n = orb<T, convAccT>(x, y, score, ori, size, desc,
                              getArray<T>(in), fast_thr, max_feat,
                              scl_fctr, levels, blur_img);

    feat.x           = getHandle(x);
    feat.y           = getHandle(y);
    feat.score       = getHandle(score);
    feat.orientation = getHandle(ori);
    feat.size        = getHandle(size);

    feat_ = getFeaturesHandle(feat);
    descriptor = getHandle<unsigned>(desc);
}

af_err af_orb(af_features* feat, af_array* desc,
              const af_array in, const float fast_thr,
              const unsigned max_feat, const float scl_fctr,
              const unsigned levels, const bool blur_img)
{
    try {
        ArrayInfo info = getInfo(in);
        af::dim4 dims  = info.dims();

        ARG_ASSERT(2, (dims[0] >= 7 && dims[1] >= 7 && dims[2] == 1 && dims[3] == 1));
        ARG_ASSERT(3, fast_thr > 0.0f);
        ARG_ASSERT(4, max_feat > 0);
        ARG_ASSERT(5, scl_fctr > 1.0f);
        ARG_ASSERT(6, levels > 0);

        dim_t in_ndims = dims.ndims();
        DIM_ASSERT(1, (in_ndims <= 3 && in_ndims >= 2));

        af_array tmp_desc;
        af_dtype type  = info.getType();
        switch(type) {
            case f32: orb<float , float >(*feat, tmp_desc, in, fast_thr, max_feat,
                                          scl_fctr, levels, blur_img); break;
            case f64: orb<double, double>(*feat, tmp_desc, in, fast_thr, max_feat,
                                          scl_fctr, levels, blur_img); break;
            default : TYPE_ERROR(1, type);
        }
        std::swap(*desc, tmp_desc);
    }
    CATCHALL;

    return AF_SUCCESS;
}
