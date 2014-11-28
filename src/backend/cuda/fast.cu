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
#include <err_cuda.hpp>
#include <handle.hpp>
#include <kernel/fast.hpp>

using af::dim4;
using af::features;

namespace cuda
{

template<typename T>
features * fast(const Array<T> &in, const float thr, const unsigned arc_length,
                const bool non_max, const float feature_ratio)
{
    const dim4 dims = in.dims();

    unsigned nfeat;
    float *x_out;
    float *y_out;
    float *score_out;

    kernel::fast<T>(&nfeat, &x_out, &y_out, &score_out, in,
                    thr, arc_length, non_max, feature_ratio);

    const dim4 out_dims(nfeat);

    Array<float> * x = createDeviceDataArray<float>(out_dims, x_out);
    Array<float> * y = createDeviceDataArray<float>(out_dims, y_out);
    Array<float> * score = createDeviceDataArray<float>(out_dims, score_out);
    Array<float> * orientation = createValueArray<float>(out_dims, 0.0f);
    Array<float> * size = createValueArray<float>(out_dims, 1.0f);

    features * feat = new features;
    feat->setNumFeatures(nfeat);
    feat->setX(getHandle<float>(*x));
    feat->setY(getHandle<float>(*y));
    feat->setScore(getHandle<float>(*score));
    feat->setOrientation(getHandle<float>(*orientation));
    feat->setSize(getHandle<float>(*size));

    return feat;
}

#define INSTANTIATE(T)\
    template features * fast<T>(const Array<T> &in, const float thr, const unsigned arc_length,  \
                                const bool non_max, const float feature_ratio);

INSTANTIATE(float )
INSTANTIATE(double)
INSTANTIATE(char  )
INSTANTIATE(int   )
INSTANTIATE(uint  )
INSTANTIATE(uchar )

}
