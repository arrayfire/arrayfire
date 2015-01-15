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
#include <kernel/orb.hpp>

using af::dim4;
using af::features;

namespace opencl
{

template<typename T, typename convAccT>
void orb(features& feat, Array<unsigned>** desc, const Array<T>& in,
         const float fast_thr, const unsigned max_feat,
         const float scl_fctr, const unsigned levels)
{
    unsigned nfeat;

    Param x;
    Param y;
    Param score;
    Param ori;
    Param size;
    Param desc_tmp;

    kernel::orb<T,convAccT>(&nfeat, x, y, score, ori, size, desc_tmp,
                            in, fast_thr, max_feat, scl_fctr, levels);

    if (nfeat == 0) {
        feat.setNumFeatures(0);
        return;
    }

    const dim4 out_dims(nfeat);
    const dim4 desc_dims(8, nfeat);

    feat.setNumFeatures(nfeat);
    feat.setX(getHandle<float>(*createParamArray<float>(x)));
    feat.setY(getHandle<float>(*createParamArray<float>(y)));
    feat.setScore(getHandle<float>(*createParamArray<float>(score)));
    feat.setOrientation(getHandle<float>(*createParamArray<float>(ori)));
    feat.setSize(getHandle<float>(*createParamArray<float>(size)));

    *desc = createParamArray<unsigned>(desc_tmp);
}

#define INSTANTIATE(T, convAccT)\
    template void orb<T, convAccT>(features& feat, Array<unsigned>** desc, const Array<T>& in,  \
                                   const float fast_thr, const unsigned max_feat,               \
                                   const float scl_fctr, const unsigned levels);

INSTANTIATE(float , float)
INSTANTIATE(double, double)
INSTANTIATE(char  , float)
INSTANTIATE(int   , float)
INSTANTIATE(uint  , float)
INSTANTIATE(uchar , float)

}
