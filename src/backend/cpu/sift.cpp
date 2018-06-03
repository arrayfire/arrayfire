/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <Array.hpp>
#include <err_cpu.hpp>
#include <resize.hpp>
#include <sort_index.hpp>
#include <convolve.hpp>
#include <math.hpp>
#include <memory.hpp>
#include <cstring>
#include <cfloat>
#include <vector>

#ifdef AF_WITH_NONFREE_SIFT
#include <kernel/sift_nonfree.hpp>
#endif

using af::dim4;

namespace cpu
{

template<typename T, typename convAccT>
unsigned sift(Array<float>& x, Array<float>& y, Array<float>& score,
              Array<float>& ori, Array<float>& size, Array<float>& desc,
              const Array<T>& in, const unsigned n_layers,
              const float contrast_thr, const float edge_thr,
              const float init_sigma, const bool double_input,
              const float img_scale, const float feature_ratio,
              const bool compute_GLOH)
{
#ifdef AF_WITH_NONFREE_SIFT
    return sift_impl<T, convAccT>(x, y, score, ori, size, desc, in, n_layers,
                                  contrast_thr, edge_thr, init_sigma, double_input,
                                  img_scale, feature_ratio, compute_GLOH);
#else
    if (compute_GLOH)
        AF_ERROR("ArrayFire was not built with nonfree support, GLOH disabled\n", AF_ERR_NONFREE);
    else
        AF_ERROR("ArrayFire was not built with nonfree support, SIFT disabled\n", AF_ERR_NONFREE);
#endif
}

#define INSTANTIATE(T, convAccT)\
    template unsigned sift<T, convAccT>(Array<float>& x, Array<float>& y,                   \
                                        Array<float>& score, Array<float>& ori,             \
                                        Array<float>& size, Array<float>& desc,             \
                                        const Array<T>& in, const unsigned n_layers,        \
                                        const float contrast_thr, const float edge_thr,     \
                                        const float init_sigma, const bool double_input,    \
                                        const float img_scale, const float feature_ratio,   \
                                        const bool compute_GLOH);

INSTANTIATE(float , float )
INSTANTIATE(double, double)

}
