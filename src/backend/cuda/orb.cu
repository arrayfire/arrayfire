/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <orb.hpp>

#include <Array.hpp>
#include <LookupTable1D.hpp>
#include <err_cuda.hpp>
#include <fast_pyramid.hpp>
#include <kernel/orb.hpp>
#include <kernel/orb_patch.hpp>
#include <af/dim4.hpp>

#include <type_traits>

using af::dim4;

namespace arrayfire {
namespace cuda {

template<typename T, typename convAccT>
unsigned orb(Array<float> &x, Array<float> &y, Array<float> &score,
             Array<float> &ori, Array<float> &size, Array<uint> &desc,
             const Array<T> &image, const float fast_thr,
             const unsigned max_feat, const float scl_fctr,
             const unsigned levels, const bool blur_img) {
    std::vector<unsigned> feat_pyr, lvl_best;
    std::vector<float> lvl_scl;
    std::vector<Array<float>> x_pyr, y_pyr;
    std::vector<Array<T>> img_pyr;

    fast_pyramid<T>(feat_pyr, x_pyr, y_pyr, lvl_best, lvl_scl, img_pyr, image,
                    fast_thr, max_feat, scl_fctr, levels, REF_PAT_SIZE);

    const size_t num_levels = feat_pyr.size();

    std::vector<float *> d_x_pyr(num_levels, nullptr),
        d_y_pyr(num_levels, nullptr);

    for (size_t i = 0; i < feat_pyr.size(); ++i) {
        if (feat_pyr[i] > 0) {
            d_x_pyr[i] = static_cast<float *>(x_pyr[i].get());
            d_y_pyr[i] = static_cast<float *>(y_pyr[i].get());
        }
    }

    unsigned nfeat_out;
    float *x_out;
    float *y_out;
    float *score_out;
    float *orientation_out;
    float *size_out;
    unsigned *desc_out;

    // TODO(pradeep) Figure out a better way to create lut Array only once
    const Array<int> lut = createHostDataArray(
        af::dim4(sizeof(d_ref_pat) / sizeof(int)), d_ref_pat);

    LookupTable1D<int> orbLUT(lut);

    kernel::orb<T, convAccT>(
        &nfeat_out, &x_out, &y_out, &score_out, &orientation_out, &size_out,
        &desc_out, feat_pyr, d_x_pyr, d_y_pyr, lvl_best, lvl_scl, img_pyr,
        fast_thr, max_feat, scl_fctr, levels, blur_img, orbLUT);

    if (nfeat_out > 0) {
        if (x_out == NULL || y_out == NULL || score_out == NULL ||
            orientation_out == NULL || size_out == NULL || desc_out == NULL) {
            AF_ERROR("orb_descriptor: feature array is null.", AF_ERR_SIZE);
        }

        const dim4 feat_dims(nfeat_out);
        const dim4 desc_dims(8, nfeat_out);

        x     = createDeviceDataArray<float>(feat_dims, x_out);
        y     = createDeviceDataArray<float>(feat_dims, y_out);
        score = createDeviceDataArray<float>(feat_dims, score_out);
        ori   = createDeviceDataArray<float>(feat_dims, orientation_out);
        size  = createDeviceDataArray<float>(feat_dims, size_out);
        desc  = createDeviceDataArray<unsigned>(desc_dims, desc_out);
    }

    return nfeat_out;
}

#define INSTANTIATE(T, convAccT)                                              \
    template unsigned orb<T, convAccT>(                                       \
        Array<float> & x, Array<float> & y, Array<float> & score,             \
        Array<float> & ori, Array<float> & size, Array<uint> & desc,          \
        const Array<T> &image, const float fast_thr, const unsigned max_feat, \
        const float scl_fctr, const unsigned levels, const bool blur_img);

INSTANTIATE(float, float)
INSTANTIATE(double, double)

}  // namespace cuda
}  // namespace arrayfire
