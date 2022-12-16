/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <err_opencl.hpp>
#include <kernel/orb.hpp>
#include <math.hpp>
#include <af/dim4.hpp>
#include <af/features.h>

using af::dim4;
using af::features;

namespace arrayfire {
namespace opencl {

template<typename T, typename convAccT>
unsigned orb(Array<float> &x_out, Array<float> &y_out, Array<float> &score_out,
             Array<float> &ori_out, Array<float> &size_out,
             Array<uint> &desc_out, const Array<T> &image, const float fast_thr,
             const unsigned max_feat, const float scl_fctr,
             const unsigned levels, const bool blur_img) {
    unsigned nfeat;

    Param x;
    Param y;
    Param score;
    Param ori;
    Param size;
    Param desc;

    kernel::orb<T, convAccT>(&nfeat, x, y, score, ori, size, desc, image,
                             fast_thr, max_feat, scl_fctr, levels, blur_img);

    if (nfeat > 0) {
        const dim4 out_dims(nfeat);
        const dim4 desc_dims(8, nfeat);

        x_out     = createParamArray<float>(x, true);
        y_out     = createParamArray<float>(y, true);
        score_out = createParamArray<float>(score, true);
        ori_out   = createParamArray<float>(ori, true);
        size_out  = createParamArray<float>(size, true);
        desc_out  = createParamArray<unsigned>(desc, true);
    }

    return nfeat;
}

#define INSTANTIATE(T, convAccT)                                              \
    template unsigned orb<T, convAccT>(                                       \
        Array<float> & x, Array<float> & y, Array<float> & score,             \
        Array<float> & ori, Array<float> & size, Array<uint> & desc,          \
        const Array<T> &image, const float fast_thr, const unsigned max_feat, \
        const float scl_fctr, const unsigned levels, const bool blur_img);

INSTANTIATE(float, float)
INSTANTIATE(double, double)

}  // namespace opencl
}  // namespace arrayfire
