/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fast.hpp>
#include <kernel/fast.hpp>

#include <Array.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <af/dim4.hpp>
#include <cmath>

#include <algorithm>
#include <cmath>
#include <cstddef>

using af::dim4;
using std::ceil;

namespace arrayfire {
namespace cpu {

template<typename T>
unsigned fast(Array<float> &x_out, Array<float> &y_out, Array<float> &score_out,
              const Array<T> &in, const float thr, const unsigned arc_length,
              const bool nonmax, const float feature_ratio,
              const unsigned edge) {
    in.eval();

    dim4 in_dims            = in.dims();
    const unsigned max_feat = ceil(in.elements() * feature_ratio);

    // Matrix containing scores for detected features, scores are stored in the
    // same coordinates as features, dimensions should be equal to in.
    Array<float> V = createEmptyArray<float>(dim4());
    if (nonmax == 1) {
        dim4 V_dims(in_dims[0], in_dims[1]);
        V = createValueArray<float>(V_dims, 0.f);
        V.eval();
    }
    getQueue().sync();

    // Arrays containing all features detected before non-maximal suppression.
    dim4 max_feat_dims(max_feat);
    Array<float> x     = createEmptyArray<float>(max_feat_dims);
    Array<float> y     = createEmptyArray<float>(max_feat_dims);
    Array<float> score = createEmptyArray<float>(max_feat_dims);

    // Feature counter
    unsigned count = 0;

    kernel::locate_features<T>(in, V, x, y, score, &count, thr, arc_length,
                               nonmax, max_feat, edge);

    // If more features than max_feat were detected, feat wasn't populated
    // with them anyway, so the real number of features will be that of
    // max_feat and not count.
    unsigned feat_found = std::min(max_feat, count);
    dim4 feat_found_dims(feat_found);

    Array<float> x_total     = createEmptyArray<float>(af::dim4());
    Array<float> y_total     = createEmptyArray<float>(af::dim4());
    Array<float> score_total = createEmptyArray<float>(af::dim4());

    if (nonmax == 1) {
        x_total     = createEmptyArray<float>(feat_found_dims);
        y_total     = createEmptyArray<float>(feat_found_dims);
        score_total = createEmptyArray<float>(feat_found_dims);

        count = 0;
        kernel::non_maximal(V, x, y, x_total, y_total, score_total, &count,
                            feat_found, edge);

        feat_found = std::min(max_feat, count);
    } else {
        x_total     = x;
        y_total     = y;
        score_total = score;
    }

    if (feat_found > 0) {
        feat_found_dims = dim4(feat_found);

        x_out     = createEmptyArray<float>(feat_found_dims);
        y_out     = createEmptyArray<float>(feat_found_dims);
        score_out = createEmptyArray<float>(feat_found_dims);

        float *x_total_ptr     = x_total.get();
        float *y_total_ptr     = y_total.get();
        float *score_total_ptr = score_total.get();

        float *x_out_ptr     = x_out.get();
        float *y_out_ptr     = y_out.get();
        float *score_out_ptr = score_out.get();

        for (size_t i = 0; i < feat_found; i++) {
            x_out_ptr[i]     = x_total_ptr[i];
            y_out_ptr[i]     = y_total_ptr[i];
            score_out_ptr[i] = score_total_ptr[i];
        }
    }

    return feat_found;
}

#define INSTANTIATE(T)                                                        \
    template unsigned fast<T>(                                                \
        Array<float> & x_out, Array<float> & y_out, Array<float> & score_out, \
        const Array<T> &in, const float thr, const unsigned arc_length,       \
        const bool nonmax, const float feature_ratio, const unsigned edge);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(char)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)

}  // namespace cpu
}  // namespace arrayfire
