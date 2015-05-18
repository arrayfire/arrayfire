/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <dispatch.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include <memory.hpp>

#include "fast.hpp"
#include "resize.hpp"

namespace cuda
{

namespace kernel
{

template<typename T>
void fast_pyramid(std::vector<unsigned>& feat_pyr,
                  std::vector<float*>& d_x_pyr,
                  std::vector<float*>& d_y_pyr,
                  std::vector<unsigned>& lvl_best,
                  std::vector<float>& lvl_scl,
                  std::vector<CParam<T> >& img_pyr,
                  CParam<T> in,
                  const float fast_thr,
                  const unsigned max_feat,
                  const float scl_fctr,
                  const unsigned levels,
                  const unsigned patch_size)
{
    unsigned min_side = std::min(in.dims[0], in.dims[1]);
    unsigned max_levels = 0;
    float scl_sum = 0.f;

    for (unsigned i = 0; i < levels; i++) {
        min_side /= scl_fctr;

        // Minimum image side for a descriptor to be computed
        if (min_side < patch_size || max_levels == levels) break;

        max_levels++;
        scl_sum += 1.f / (float)std::pow(scl_fctr,(float)i);
    }

    // Compute number of features to keep for each level
    lvl_best.resize(max_levels);
    lvl_scl.resize(max_levels);
    unsigned feat_sum = 0;
    for (unsigned i = 0; i < max_levels-1; i++) {
        float scl = (float)std::pow(scl_fctr,(float)i);
        lvl_scl[i] = scl;

        lvl_best[i] = ceil((max_feat / scl_sum) / lvl_scl[i]);
        feat_sum += lvl_best[i];
    }
    lvl_scl[max_levels-1] = (float)std::pow(scl_fctr,(float)max_levels-1);
    lvl_best[max_levels-1] = max_feat - feat_sum;

    // Hold multi-scale image pyramids
    img_pyr.reserve(max_levels);

    // Create multi-scale image pyramid
    for (unsigned i = 0; i < max_levels; i++) {
        if (i == 0) {
            // First level is used in its original size
            img_pyr[i].ptr = in.ptr;
            for (int k = 0; k < 4; k++) {
                img_pyr[i].dims[k] = in.dims[k];
                img_pyr[i].strides[k] = in.strides[k];
            }
        }
        else {
            // Resize previous level image to current level dimensions
            Param<T> lvl_img;
            lvl_img.dims[0] = round(in.dims[0] / lvl_scl[i]);
            lvl_img.dims[1] = round(in.dims[1] / lvl_scl[i]);
            lvl_img.strides[0] = 1;
            lvl_img.strides[1] = lvl_img.dims[0] * lvl_img.strides[0];

            for (int k = 2; k < 4; k++) {
                lvl_img.dims[k] = 1;
                lvl_img.strides[k] = lvl_img.dims[k - 1] * lvl_img.strides[k - 1];
            }

            int lvl_elem = lvl_img.strides[3] * lvl_img.dims[3];
            lvl_img.ptr = memAlloc<T>(lvl_elem);

            resize<T, AF_INTERP_BILINEAR>(lvl_img, img_pyr[i-1]);

            img_pyr[i].ptr = lvl_img.ptr;
            for (int k = 0; k < 4; k++) {
                img_pyr[i].dims[k] = lvl_img.dims[k];
                img_pyr[i].strides[k] = lvl_img.strides[k];
            }
        }
    }

    feat_pyr.resize(max_levels);
    d_x_pyr.resize(max_levels);
    d_y_pyr.resize(max_levels);

    for (unsigned i = 0; i < max_levels; i++) {
        unsigned lvl_feat = 0;
        float* d_x_feat = NULL;
        float* d_y_feat = NULL;
        float* d_score_feat = NULL;

        // Round feature size to nearest odd integer
        float size = 2.f * floor(patch_size / 2.f) + 1.f;

        // Avoid keeping features that are too wide and might not fit the image,
        // sqrt(2.f) is the radius when angle is 45 degrees and represents
        // widest case possible
        unsigned edge = ceil(size * sqrt(2.f) / 2.f);

        // Detects FAST features
        fast(&lvl_feat, &d_x_feat, &d_y_feat, &d_score_feat,
             img_pyr[i], fast_thr, 9, 1, 0.15f, edge);

        // FAST score is not used
        memFree(d_score_feat);

        if (lvl_feat == 0) {
            feat_pyr[i] = 0;
            d_x_pyr[i] = NULL;
            d_x_pyr[i] = NULL;
        }
        else {
            feat_pyr[i] = lvl_feat;
            d_x_pyr[i] = d_x_feat;
            d_y_pyr[i] = d_y_feat;
        }
    }
}

} // namespace kernel

} // namespace cuda
