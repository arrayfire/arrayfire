/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/features.h>
#include <Array.hpp>
#include <err_opencl.hpp>
#include <math.hpp>

#ifdef AF_WITH_NONFREE_SIFT
#include <kernel/sift_nonfree.hpp>
#endif

using af::dim4;
using af::features;

namespace opencl
{

template<typename T, typename convAccT>
unsigned sift(Array<float>& x_out, Array<float>& y_out, Array<float>& score_out,
              Array<float>& ori_out, Array<float>& size_out, Array<float>& desc_out,
              const Array<T>& in, const unsigned n_layers,
              const float contrast_thr, const float edge_thr,
              const float init_sigma, const bool double_input,
              const float img_scale, const float feature_ratio,
              const bool compute_GLOH)
{
#ifdef AF_WITH_NONFREE_SIFT
    unsigned nfeat_out;
    unsigned desc_len;

    Param x;
    Param y;
    Param score;
    Param ori;
    Param size;
    Param desc;

    kernel::sift<T,convAccT>(&nfeat_out, &desc_len, x, y, score, ori, size, desc,
                             in, n_layers, contrast_thr, edge_thr, init_sigma,
                             double_input, img_scale, feature_ratio, compute_GLOH);

    if (nfeat_out > 0) {
        const dim4 out_dims(nfeat_out);
        const dim4 desc_dims(desc_len, nfeat_out);

        x_out     = createParamArray<float>(x, true);
        y_out     = createParamArray<float>(y, true);
        score_out = createParamArray<float>(score, true);
        ori_out   = createParamArray<float>(ori, true);
        size_out  = createParamArray<float>(size, true);
        desc_out  = createParamArray<float>(desc, true);
    }

    return nfeat_out;
#else
    if (compute_GLOH)
        AF_ERROR("ArrayFire was not built with nonfree support, GLOH disabled\n", AF_ERR_NONFREE);
    else
        AF_ERROR("ArrayFire was not built with nonfree support, SIFT disabled\n", AF_ERR_NONFREE);
#endif
}


#define INSTANTIATE(T, convAccT)                                                            \
    template unsigned sift<T, convAccT>(Array<float>& x_out, Array<float>& y_out,           \
                                        Array<float>& score_out, Array<float>& ori_out,     \
                                        Array<float>& size_out, Array<float>& desc_out,     \
                                        const Array<T>& in, const unsigned n_layers,        \
                                        const float contrast_thr, const float edge_thr,     \
                                        const float init_sigma, const bool double_input,    \
                                        const float img_scale, const float feature_ratio,   \
                                        const bool compute_GLOH);

INSTANTIATE(float , float )
INSTANTIATE(double, double)

}
