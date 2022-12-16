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
#include <kernel/fast.hpp>
#include <af/dim4.hpp>
#include <af/features.h>

using af::dim4;
using af::features;

namespace arrayfire {
namespace opencl {

template<typename T>
unsigned fast(Array<float> &x_out, Array<float> &y_out, Array<float> &score_out,
              const Array<T> &in, const float thr, const unsigned arc_length,
              const bool non_max, const float feature_ratio,
              const unsigned edge) {
    unsigned nfeat;

    Param x;
    Param y;
    Param score;

    kernel::fast<T>(arc_length, &nfeat, x, y, score, in, thr, feature_ratio,
                    edge, non_max);

    if (nfeat > 0) {
        x_out     = createParamArray<float>(x, true);
        y_out     = createParamArray<float>(y, true);
        score_out = createParamArray<float>(score, true);
    }

    return nfeat;
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

}  // namespace opencl
}  // namespace arrayfire
