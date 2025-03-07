/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fast_pyramid.hpp>

#include <Array.hpp>
#include <err_cuda.hpp>
#include <fast.hpp>
#include <resize.hpp>
#include <af/dim4.hpp>

using af::dim4;
using std::vector;

namespace arrayfire {
namespace cuda {

template<typename T>
void fast_pyramid(vector<unsigned> &feat_pyr, vector<Array<float>> &x_pyr,
                  vector<Array<float>> &y_pyr, vector<unsigned> &lvl_best,
                  vector<float> &lvl_scl, vector<Array<T>> &img_pyr,
                  const Array<T> &in, const float fast_thr,
                  const unsigned max_feat, const float scl_fctr,
                  const unsigned levels, const unsigned patch_size) {
    dim4 indims         = in.dims();
    unsigned min_side   = std::min(indims[0], indims[1]);
    unsigned max_levels = 0;
    float scl_sum       = 0.f;

    for (unsigned i = 0; i < levels; i++) {
        min_side /= scl_fctr;

        // Minimum image side for a descriptor to be computed
        if (min_side < patch_size || max_levels == levels) { break; }

        max_levels++;
        scl_sum += 1.f / std::pow(scl_fctr, static_cast<float>(i));
    }

    // Compute number of features to keep for each level
    lvl_best.resize(max_levels);
    lvl_scl.resize(max_levels);
    unsigned feat_sum = 0;
    for (unsigned i = 0; i < max_levels - 1; i++) {
        auto scl   = std::pow(scl_fctr, static_cast<float>(i));
        lvl_scl[i] = scl;

        lvl_best[i] = ceil((max_feat / scl_sum) / lvl_scl[i]);
        feat_sum += lvl_best[i];
    }
    lvl_scl[max_levels - 1] =
        std::pow(scl_fctr, static_cast<float>(max_levels) - 1);
    lvl_best[max_levels - 1] = max_feat - feat_sum;

    // Hold multi-scale image pyramids
    static const dim4 dims0;
    static const CParam<T> emptyCParam(NULL, dims0.get(), dims0.get());

    img_pyr.reserve(max_levels);

    // Create multi-scale image pyramid
    for (unsigned i = 0; i < max_levels; i++) {
        if (i == 0) {
            // First level is used in its original size
            img_pyr.push_back(in);
        } else {
            // Resize previous level image to current level dimensions
            dim4 dims(round(indims[0] / lvl_scl[i]),
                      round(indims[1] / lvl_scl[i]));

            img_pyr.push_back(createEmptyArray<T>(dims));
            img_pyr[i] =
                resize(img_pyr[i - 1], dims[0], dims[1], AF_INTERP_BILINEAR);
        }
    }

    feat_pyr.resize(max_levels);

    // Round feature size to nearest odd integer
    float size = 2.f * floor(patch_size / 2.f) + 1.f;

    // Avoid keeping features that are too wide and might not fit the image,
    // sqrt(2.f) is the radius when angle is 45 degrees and represents
    // widest case possible
    unsigned edge = ceil(size * sqrt(2.f) / 2.f);

    for (unsigned i = 0; i < max_levels; i++) {
        Array<float> x_out     = createEmptyArray<float>(dim4());
        Array<float> y_out     = createEmptyArray<float>(dim4());
        Array<float> score_out = createEmptyArray<float>(dim4());

        unsigned lvl_feat = fast(x_out, y_out, score_out, img_pyr[i], fast_thr,
                                 9, 1, 0.14f, edge);

        if (lvl_feat > 0) {
            feat_pyr[i] = lvl_feat;
            x_pyr.push_back(x_out);
            y_pyr.push_back(y_out);
        } else {
            feat_pyr[i] = 0;
        }
    }
}

#define INSTANTIATE(T)                                                      \
    template void fast_pyramid<T>(                                          \
        vector<unsigned> &, vector<Array<float>> &, vector<Array<float>> &, \
        vector<unsigned> &, vector<float> &, vector<Array<T>> &,            \
        const Array<T> &, const float, const unsigned, const float,         \
        const unsigned, const unsigned);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(char)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)

}  // namespace cuda
}  // namespace arrayfire
