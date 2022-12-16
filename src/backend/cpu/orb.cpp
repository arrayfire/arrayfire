/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <convolve.hpp>
#include <fast.hpp>
#include <kernel/orb.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <resize.hpp>
#include <sort_index.hpp>
#include <af/dim4.hpp>

#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

using af::dim4;
using std::ceil;
using std::floor;
using std::function;
using std::min;
using std::move;
using std::pow;
using std::round;
using std::sqrt;
using std::unique_ptr;
using std::vector;

namespace arrayfire {
namespace cpu {

template<typename T, typename convAccT>
unsigned orb(Array<float>& x, Array<float>& y, Array<float>& score,
             Array<float>& ori, Array<float>& size, Array<uint>& desc,
             const Array<T>& image, const float fast_thr,
             const unsigned max_feat, const float scl_fctr,
             const unsigned levels, const bool blur_img) {
    image.eval();
    getQueue().sync();

    float patch_size = REF_PAT_SIZE;

    const dim4& idims   = image.dims();
    float min_side      = min(idims[0], idims[1]);
    unsigned max_levels = 0;
    float scl_sum       = 0.f;

    for (unsigned i = 0; i < levels; i++) {
        min_side /= scl_fctr;

        // Minimum image side for a descriptor to be computed
        if (min_side < patch_size || max_levels == levels) { break; }

        max_levels++;
        scl_sum += 1.f / pow(scl_fctr, static_cast<float>(i));
    }

    vector<unique_ptr<float[], function<void(float*)>>> h_x_pyr(max_levels);
    vector<unique_ptr<float[], function<void(float*)>>> h_y_pyr(max_levels);
    vector<unique_ptr<float[], function<void(float*)>>> h_score_pyr(max_levels);
    vector<unique_ptr<float[], function<void(float*)>>> h_ori_pyr(max_levels);
    vector<unique_ptr<float[], function<void(float*)>>> h_size_pyr(max_levels);
    vector<unique_ptr<unsigned[], function<void(unsigned*)>>> h_desc_pyr(
        max_levels);

    vector<unsigned> feat_pyr(max_levels);
    unsigned total_feat = 0;

    // Compute number of features to keep for each level
    vector<unsigned> lvl_best(max_levels);
    unsigned feat_sum = 0;
    for (unsigned i = 0; i < max_levels - 1; i++) {
        auto lvl_scl = pow(scl_fctr, static_cast<float>(i));
        lvl_best[i]  = ceil((static_cast<float>(max_feat) / scl_sum) / lvl_scl);
        feat_sum += lvl_best[i];
    }
    lvl_best[max_levels - 1] = max_feat - feat_sum;

    // Maintain a reference to previous level image
    Array<T> prev_img = createEmptyArray<T>(dim4());
    dim4 prev_ldims;

    dim4 gauss_dims(9);
    unique_ptr<T[], function<void(T*)>> h_gauss;
    Array<T> gauss_filter = createEmptyArray<T>(dim4());

    for (unsigned i = 0; i < max_levels; i++) {
        dim4 ldims;
        const auto lvl_scl = pow(scl_fctr, static_cast<float>(i));
        Array<T> lvl_img   = createEmptyArray<T>(dim4());

        if (i == 0) {
            // First level is used in its original size
            lvl_img = image;
            ldims   = image.dims();

            prev_img   = image;
            prev_ldims = image.dims();
        } else {
            // Resize previous level image to current level dimensions
            ldims[0] = round(idims[0] / lvl_scl);
            ldims[1] = round(idims[1] / lvl_scl);

            lvl_img =
                resize<T>(prev_img, ldims[0], ldims[1], AF_INTERP_BILINEAR);

            prev_img   = lvl_img;
            prev_ldims = lvl_img.dims();
        }
        prev_img.eval();
        lvl_img.eval();
        getQueue().sync();

        Array<float> x_feat     = createEmptyArray<float>(dim4());
        Array<float> y_feat     = createEmptyArray<float>(dim4());
        Array<float> score_feat = createEmptyArray<float>(dim4());

        // Round feature size to nearest odd integer
        float size = 2.f * floor(static_cast<float>(patch_size) / 2.f) + 1.f;

        // Avoid keeping features that might be too wide and might not fit on
        // the image, sqrt(2.f) is the radius when angle is 45 degrees and
        // represents widest case possible
        unsigned edge = ceil(size * sqrt(2.f) / 2.f);

        unsigned lvl_feat = fast(x_feat, y_feat, score_feat, lvl_img, fast_thr,
                                 9, 1, 0.15f, edge);

        if (lvl_feat == 0) { continue; }

        float* h_x_feat = x_feat.get();
        float* h_y_feat = y_feat.get();

        auto h_x_harris     = memAlloc<float>(lvl_feat);
        auto h_y_harris     = memAlloc<float>(lvl_feat);
        auto h_score_harris = memAlloc<float>(lvl_feat);

        // Calculate Harris responses
        // Good block_size >= 7 (must be an odd number)
        unsigned usable_feat = 0;
        kernel::harris_response<T, false>(
            h_x_harris.get(), h_y_harris.get(), h_score_harris.get(), nullptr,
            h_x_feat, h_y_feat, nullptr, lvl_feat, &usable_feat, lvl_img, 7,
            0.04f, patch_size);

        if (usable_feat == 0) { continue; }

        // Sort features according to Harris responses
        af::dim4 usable_feat_dims(usable_feat);
        Array<float> score_harris = createDeviceDataArray<float>(
            usable_feat_dims, h_score_harris.get());
        Array<float> harris_sorted = createEmptyArray<float>(af::dim4());
        Array<unsigned> harris_idx = createEmptyArray<unsigned>(af::dim4());

        sort_index<float>(harris_sorted, harris_idx, score_harris, 0, false);
        getQueue().sync();

        usable_feat = min(usable_feat, lvl_best[i]);

        if (usable_feat == 0) {
            h_score_harris.release();
            continue;
        }

        auto h_x_lvl     = memAlloc<float>(usable_feat);
        auto h_y_lvl     = memAlloc<float>(usable_feat);
        auto h_score_lvl = memAlloc<float>(usable_feat);

        // Keep only features with higher Harris responses
        kernel::keep_features<T>(h_x_lvl.get(), h_y_lvl.get(),
                                 h_score_lvl.get(), nullptr, h_x_harris.get(),
                                 h_y_harris.get(), harris_sorted.get(),
                                 harris_idx.get(), nullptr, usable_feat);

        auto h_ori_lvl  = memAlloc<float>(usable_feat);
        auto h_size_lvl = memAlloc<float>(usable_feat);

        // Compute orientation of features
        kernel::centroid_angle<T>(h_x_lvl.get(), h_y_lvl.get(), h_ori_lvl.get(),
                                  usable_feat, lvl_img, patch_size);

        Array<T> lvl_filt = createEmptyArray<T>(dim4());

        if (blur_img) {
            // Calculate a separable Gaussian kernel, if one is not already
            // stored
            if (!h_gauss) {
                h_gauss = memAlloc<T>(gauss_dims[0]);
                gaussian1D(h_gauss.get(), gauss_dims[0], 2.f);
                gauss_filter =
                    createDeviceDataArray<T>(gauss_dims, h_gauss.get());
                gauss_filter.eval();
            }

            // Filter level image with Gaussian kernel to reduce noise
            // sensitivity
            lvl_filt = convolve2<T, convAccT>(lvl_img, gauss_filter,
                                              gauss_filter, false);
        }
        lvl_filt.eval();
        getQueue().sync();

        // Compute ORB descriptors
        auto h_desc_lvl = memAlloc<unsigned>(usable_feat * 8);
        memset(h_desc_lvl.get(), 0, usable_feat * 8 * sizeof(unsigned));
        if (blur_img) {
            kernel::extract_orb<T>(h_desc_lvl.get(), usable_feat, h_x_lvl.get(),
                                   h_y_lvl.get(), h_ori_lvl.get(),
                                   h_size_lvl.get(), lvl_filt, lvl_scl,
                                   patch_size);
        } else {
            kernel::extract_orb<T>(h_desc_lvl.get(), usable_feat, h_x_lvl.get(),
                                   h_y_lvl.get(), h_ori_lvl.get(),
                                   h_size_lvl.get(), lvl_img, lvl_scl,
                                   patch_size);
        }

        // Store results to pyramids
        total_feat += usable_feat;
        feat_pyr[i]    = usable_feat;
        h_x_pyr[i]     = move(h_x_lvl);
        h_y_pyr[i]     = move(h_y_lvl);
        h_score_pyr[i] = move(h_score_lvl);
        h_ori_pyr[i]   = move(h_ori_lvl);
        h_size_pyr[i]  = move(h_size_lvl);
        h_desc_pyr[i]  = move(h_desc_lvl);
        h_score_harris.release();
        h_gauss.release();
    }

    if (total_feat > 0) {
        // Allocate feature Arrays
        const af::dim4 total_feat_dims(total_feat);
        const af::dim4 desc_dims(8, total_feat);

        x     = createEmptyArray<float>(total_feat_dims);
        y     = createEmptyArray<float>(total_feat_dims);
        score = createEmptyArray<float>(total_feat_dims);
        ori   = createEmptyArray<float>(total_feat_dims);
        size  = createEmptyArray<float>(total_feat_dims);
        desc  = createEmptyArray<uint>(desc_dims);

        float* h_x     = x.get();
        float* h_y     = y.get();
        float* h_score = score.get();
        float* h_ori   = ori.get();
        float* h_size  = size.get();

        unsigned* h_desc = desc.get();

        unsigned offset = 0;
        for (unsigned i = 0; i < max_levels; i++) {
            if (feat_pyr[i] == 0) { continue; }

            if (i > 0) { offset += feat_pyr[i - 1]; }

            memcpy(h_x + offset, h_x_pyr[i].get(), feat_pyr[i] * sizeof(float));
            memcpy(h_y + offset, h_y_pyr[i].get(), feat_pyr[i] * sizeof(float));
            memcpy(h_score + offset, h_score_pyr[i].get(),
                   feat_pyr[i] * sizeof(float));
            memcpy(h_ori + offset, h_ori_pyr[i].get(),
                   feat_pyr[i] * sizeof(float));
            memcpy(h_size + offset, h_size_pyr[i].get(),
                   feat_pyr[i] * sizeof(float));

            memcpy(h_desc + (offset * 8), h_desc_pyr[i].get(),
                   feat_pyr[i] * 8 * sizeof(unsigned));
        }
    }

    return total_feat;
}

#define INSTANTIATE(T, convAccT)                                              \
    template unsigned orb<T, convAccT>(                                       \
        Array<float> & x, Array<float> & y, Array<float> & score,             \
        Array<float> & ori, Array<float> & size, Array<uint> & desc,          \
        const Array<T>& image, const float fast_thr, const unsigned max_feat, \
        const float scl_fctr, const unsigned levels, const bool blur_img);

INSTANTIATE(float, float)
INSTANTIATE(double, double)

}  // namespace cpu
}  // namespace arrayfire
