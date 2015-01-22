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
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <err_cuda.hpp>
#include <handle.hpp>
#include <fast_pyramid.hpp>
#include <kernel/orb.hpp>
#include <kernel/orb_patch.hpp>

using af::dim4;
using af::features;

namespace cuda
{

template<typename T, typename convAccT>
void orb(features& feat, Array<unsigned>** desc, const Array<T>& image,
         const float fast_thr, const unsigned max_feat,
         const float scl_fctr, const unsigned levels)
{
    const dim4 dims = image.dims();

    std::vector<unsigned> feat_pyr, lvl_best;
    std::vector<float> lvl_scl;
    std::vector<float*> d_x_pyr, d_y_pyr;
    std::vector<CParam<T> > img_pyr;

    fast_pyramid<T>(feat_pyr, d_x_pyr, d_y_pyr, lvl_best, lvl_scl, img_pyr,
                    image, fast_thr, max_feat, scl_fctr, levels, REF_PAT_SIZE);

    unsigned nfeat_out;
    float *x_out;
    float *y_out;
    float *score_out;
    float *orientation_out;
    float *size_out;
    unsigned *desc_out;

    kernel::orb<T, convAccT>(&nfeat_out, &x_out, &y_out, &score_out, &orientation_out, &size_out,
                             &desc_out, feat_pyr, d_x_pyr, d_y_pyr, lvl_best, lvl_scl, img_pyr,
                             fast_thr, max_feat, scl_fctr, levels);

    if (nfeat_out == 0) {
        feat.setNumFeatures(0);
        feat.setX(getHandle<float>(*createEmptyArray<float>(af::dim4())));
        feat.setY(getHandle<float>(*createEmptyArray<float>(af::dim4())));
        feat.setScore(getHandle<float>(*createEmptyArray<float>(af::dim4())));
        feat.setOrientation(getHandle<float>(*createEmptyArray<float>(af::dim4())));
        feat.setSize(getHandle<float>(*createEmptyArray<float>(af::dim4())));
        *desc = createEmptyArray<unsigned>(af::dim4());
        return;
    }
    if (x_out == NULL || y_out == NULL || score_out == NULL || orientation_out == NULL ||
        size_out == NULL || desc_out == NULL) {
        AF_ERROR("orb_descriptor: feature array is null.", AF_ERR_SIZE);
    }

    const dim4 feat_dims(nfeat_out);
    const dim4 desc_dims(8, nfeat_out);

    Array<float> * x = createDeviceDataArray<float>(feat_dims, x_out);
    Array<float> * y = createDeviceDataArray<float>(feat_dims, y_out);
    Array<float> * score = createDeviceDataArray<float>(feat_dims, score_out);
    Array<float> * orientation = createDeviceDataArray<float>(feat_dims, orientation_out);
    Array<float> * size = createDeviceDataArray<float>(feat_dims, size_out);
    *desc = createDeviceDataArray<unsigned>(desc_dims, desc_out);

    feat.setNumFeatures(nfeat_out);
    feat.setX(getHandle<float>(*x));
    feat.setY(getHandle<float>(*y));
    feat.setScore(getHandle<float>(*score));
    feat.setOrientation(getHandle<float>(*orientation));
    feat.setSize(getHandle<float>(*size));
}

#define INSTANTIATE(T, convAccT)\
    template void orb<T, convAccT>(features &feat, Array<unsigned>** desc, const Array<T>& image,   \
                                   const float fast_thr, const unsigned max_feat,                   \
                                   const float scl_fctr, const unsigned levels);

INSTANTIATE(float , float )
INSTANTIATE(double, double)

}
