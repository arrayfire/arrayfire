/*******************************************************
 * Copyright (c) 2015, ArrayFire
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
#include <err_opencl.hpp>
#include <handle.hpp>
#include <kernel/harris.hpp>

using af::dim4;
using af::features;

namespace opencl
{

template<typename T, typename convAccT>
unsigned harris(Array<float> &x_out, Array<float> &y_out, Array<float> &score_out,
                const Array<T> &in, const unsigned max_corners, const float min_response,
                const float sigma, const unsigned filter_len, const float k_thr)
{
    unsigned nfeat;

    Param x;
    Param y;
    Param score;

    kernel::harris<T, convAccT>(&nfeat, x, y, score, in,
                                max_corners, min_response,
                                sigma, filter_len, k_thr);

    if (nfeat > 0) {
        x_out = createParamArray<float>(x);
        y_out = createParamArray<float>(y);
        score_out = createParamArray<float>(score);
    }

    return nfeat;
}

#define INSTANTIATE(T, convAccT)                                                                                    \
    template unsigned harris<T, convAccT>(Array<float> &x_out, Array<float> &y_out, Array<float> &score_out,        \
                                          const Array<T> &in, const unsigned max_corners, const float min_response, \
                                          const float sigma, const unsigned filter_len, const float k_thr);

INSTANTIATE(double, double)
INSTANTIATE(float , float)

}
