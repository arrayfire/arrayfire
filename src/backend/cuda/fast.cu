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
unsigned fast(Array<float> &x_out, Array<float> &y_out, Array<float> &score_out,
              const Array<T> &in, const float thr, const unsigned arc_length,
              const bool non_max, const float feature_ratio, const unsigned edge)
{
    const dim4 dims = in.dims();

    unsigned nfeat;
    float *d_x_out;
    float *d_y_out;
    float *d_score_out;

    kernel::fast<T>(&nfeat, &d_x_out, &d_y_out, &d_score_out, in,
                    thr, arc_length, non_max, feature_ratio, edge);

    if (nfeat > 0) {
        const dim4 out_dims(nfeat);

        x_out = createDeviceDataArray<float>(out_dims, d_x_out);
        y_out = createDeviceDataArray<float>(out_dims, d_y_out);
        score_out = createDeviceDataArray<float>(out_dims, d_score_out);
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
