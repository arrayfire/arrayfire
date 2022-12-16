/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <sift.hpp>

#include <kernel/sift.hpp>

using af::dim4;
using af::features;

namespace arrayfire {
namespace cuda {

template<typename T, typename convAccT>
unsigned sift(Array<float>& x, Array<float>& y, Array<float>& score,
              Array<float>& ori, Array<float>& size, Array<float>& desc,
              const Array<T>& in, const unsigned n_layers,
              const float contrast_thr, const float edge_thr,
              const float init_sigma, const bool double_input,
              const float img_scale, const float feature_ratio,
              const bool compute_GLOH) {
    unsigned nfeat_out;
    unsigned desc_len;
    float* x_out;
    float* y_out;
    float* score_out;
    float* orientation_out;
    float* size_out;
    float* desc_out;

    kernel::sift<T, convAccT>(
        &nfeat_out, &desc_len, &x_out, &y_out, &score_out, &orientation_out,
        &size_out, &desc_out, in, n_layers, contrast_thr, edge_thr, init_sigma,
        double_input, img_scale, feature_ratio, compute_GLOH);

    if (nfeat_out > 0) {
        if (x_out == NULL || y_out == NULL || score_out == NULL ||
            orientation_out == NULL || size_out == NULL || desc_out == NULL) {
            AF_ERROR("sift: feature array is null.", AF_ERR_SIZE);
        }

        const dim4 feat_dims(nfeat_out);
        const dim4 desc_dims(desc_len, nfeat_out);

        x     = createDeviceDataArray<float>(feat_dims, x_out);
        y     = createDeviceDataArray<float>(feat_dims, y_out);
        score = createDeviceDataArray<float>(feat_dims, score_out);
        ori   = createDeviceDataArray<float>(feat_dims, orientation_out);
        size  = createDeviceDataArray<float>(feat_dims, size_out);
        desc  = createDeviceDataArray<float>(desc_dims, desc_out);
    }

    return nfeat_out;
}

#define INSTANTIATE(T, convAccT)                                               \
    template unsigned sift<T, convAccT>(                                       \
        Array<float> & x, Array<float> & y, Array<float> & score,              \
        Array<float> & ori, Array<float> & size, Array<float> & desc,          \
        const Array<T>& in, const unsigned n_layers, const float contrast_thr, \
        const float edge_thr, const float init_sigma, const bool double_input, \
        const float img_scale, const float feature_ratio,                      \
        const bool compute_GLOH);

INSTANTIATE(float, float)
INSTANTIATE(double, double)

}  // namespace cuda
}  // namespace arrayfire
