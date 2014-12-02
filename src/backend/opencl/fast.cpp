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
features fast(const Array<T> &in, const float thr, const unsigned arc_length,
              const bool nonmax, const float feature_ratio)
{
    unsigned nfeat;

    Param x;
    Param y;
    Param score;

    if (!nonmax) {
        switch (arc_length) {
        case 9:
            kernel::fast<T,  9, 0>(&nfeat, x, y, score, in,
                                   thr, feature_ratio);
            break;
        case 10:
            kernel::fast<T, 10, 0>(&nfeat, x, y, score, in,
                                   thr, feature_ratio);
            break;
        case 11:
            kernel::fast<T, 11, 0>(&nfeat, x, y, score, in,
                                   thr, feature_ratio);
            break;
        case 12:
            kernel::fast<T, 12, 0>(&nfeat, x, y, score, in,
                                   thr, feature_ratio);
            break;
        case 13:
            kernel::fast<T, 13, 0>(&nfeat, x, y, score, in,
                                   thr, feature_ratio);
            break;
        case 14:
            kernel::fast<T, 14, 0>(&nfeat, x, y, score, in,
                                   thr, feature_ratio);
            break;
        case 15:
            kernel::fast<T, 15, 0>(&nfeat, x, y, score, in,
                                   thr, feature_ratio);
            break;
        case 16:
            kernel::fast<T, 16, 0>(&nfeat, x, y, score, in,
                                   thr, feature_ratio);
            break;
        }
    } else {
        switch (arc_length) {
        case 9:
            kernel::fast<T,  9, 1>(&nfeat, x, y, score, in,
                                   thr, feature_ratio);
            break;
        case 10:
            kernel::fast<T, 10, 1>(&nfeat, x, y, score, in,
                                   thr, feature_ratio);
            break;
        case 11:
            kernel::fast<T, 11, 1>(&nfeat, x, y, score, in,
                                   thr, feature_ratio);
            break;
        case 12:
            kernel::fast<T, 12, 1>(&nfeat, x, y, score, in,
                                   thr, feature_ratio);
            break;
        case 13:
            kernel::fast<T, 13, 1>(&nfeat, x, y, score, in,
                                   thr, feature_ratio);
            break;
        case 14:
            kernel::fast<T, 14, 1>(&nfeat, x, y, score, in,
                                   thr, feature_ratio);
            break;
        case 15:
            kernel::fast<T, 15, 1>(&nfeat, x, y, score, in,
                                   thr, feature_ratio);
            break;
        case 16:
            kernel::fast<T, 16, 1>(&nfeat, x, y, score, in,
                                   thr, feature_ratio);
            break;
        }
    }

    const dim4 out_dims(nfeat);

    features feat;
    feat.setNumFeatures(nfeat);
    feat.setX(getHandle<float>(*createParamArray<float>(x)));
    feat.setY(getHandle<float>(*createParamArray<float>(y)));
    feat.setScore(getHandle<float>(*createParamArray<float>(score)));
    feat.setOrientation(getHandle<float>(*createValueArray<float>(out_dims, 0.0f)));
    feat.setSize(getHandle<float>(*createValueArray<float>(out_dims, 1.0f)));

    return feat;
}

#define INSTANTIATE(T)\
    template features fast<T>(const Array<T> &in, const float thr, const unsigned arc_length, \
                              const bool non_max, const float feature_ratio);

INSTANTIATE(float )
INSTANTIATE(double)
INSTANTIATE(char  )
INSTANTIATE(int   )
INSTANTIATE(uint  )
INSTANTIATE(uchar )

}
