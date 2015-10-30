/*******************************************************
 * Copyright (c) 2014, ArrayFire
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
#include <kernel/fast.hpp>

using af::dim4;
using af::features;

namespace opencl
{

template<typename T>
unsigned fast(Array<float> &x_out, Array<float> &y_out, Array<float> &score_out,
              const Array<T> &in, const float thr, const unsigned arc_length,
              const bool non_max, const float feature_ratio, const unsigned edge)
{
    unsigned nfeat;

    Param x;
    Param y;
    Param score;

    kernel::fast_dispatch<T>(arc_length, non_max,
                             &nfeat, x, y, score, in,
                             thr, feature_ratio, edge);

    if (nfeat > 0) {
        x_out = createParamArray<float>(x);
        y_out = createParamArray<float>(y);
        score_out = createParamArray<float>(score);
    }

    return nfeat;
}

#define INSTANTIATE(T)                                                                              \
    template unsigned fast<T>(Array<float> &x_out, Array<float> &y_out, Array<float> &score_out,    \
                              const Array<T> &in, const float thr, const unsigned arc_length,       \
                              const bool nonmax, const float feature_ratio, const unsigned edge);

INSTANTIATE(float )
INSTANTIATE(double)
INSTANTIATE(char  )
INSTANTIATE(int   )
INSTANTIATE(uint  )
INSTANTIATE(uchar )
INSTANTIATE(short )
INSTANTIATE(ushort)

}
