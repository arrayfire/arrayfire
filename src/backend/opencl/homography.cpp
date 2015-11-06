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
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <err_opencl.hpp>
#include <handle.hpp>
#include <arith.hpp>
#include <random.hpp>
#include <kernel/homography.hpp>
#include <algorithm>

#include <iostream>
#include <cfloat>

using af::dim4;

namespace opencl
{

#define RANSACConfidence 0.99f
#define LMEDSConfidence 0.99f
#define LMEDSOutlierRatio 0.4f

template<typename T>
int homography(Array<T> &bestH,
               const Array<float> &x_src,
               const Array<float> &y_src,
               const Array<float> &x_dst,
               const Array<float> &y_dst,
               const af_homography_type htype,
               const float inlier_thr,
               const unsigned iterations)
{
    const af::dim4 idims = x_src.dims();
    const unsigned nsamples = idims[0];

    unsigned iter = iterations;
    Array<float> err = createEmptyArray<float>(af::dim4());
    if (htype == AF_HOMOGRAPHY_LMEDS) {
        iter = ::std::min(iter, (unsigned)(log(1.f - LMEDSConfidence) / log(1.f - pow(1.f - LMEDSOutlierRatio, 4.f))));
        err = createValueArray<float>(af::dim4(nsamples, iter), FLT_MAX);
    }
    else {
        // Avoid passing "null" cl_mem object to kernels
        err = createEmptyArray<float>(af::dim4(1));
    }

    const size_t iter_sz = divup(iter, 256) * 256;

    af::dim4 rdims(4, iter_sz);
    Array<float> frnd = randu<float>(rdims);
    Array<float> fctr = createValueArray<float>(rdims, (float)nsamples);
    Array<float> rnd = arithOp<float, af_mul_t>(frnd, fctr, rdims);

    Array<T> tmpH = createValueArray<T>(af::dim4(9, iter_sz), (T)0);
    Array<T> tmpA = createValueArray<T>(af::dim4(9, 9, iter_sz), (T)0);
    Array<T> tmpV = createValueArray<T>(af::dim4(9, 9, iter_sz), (T)0);

    bestH = createValueArray<T>(af::dim4(3, 3), (T)0);
    switch (htype) {
    case AF_HOMOGRAPHY_RANSAC:
        return kernel::computeH<T, AF_HOMOGRAPHY_RANSAC>(bestH, tmpH, tmpA, tmpV, err,
                                              x_src, y_src, x_dst, y_dst,
                                              rnd, iter, nsamples, inlier_thr);
        break;
    case AF_HOMOGRAPHY_LMEDS:
        return kernel::computeH<T, AF_HOMOGRAPHY_LMEDS> (bestH, tmpH, tmpA, tmpV, err,
                                              x_src, y_src, x_dst, y_dst,
                                              rnd, iter, nsamples, inlier_thr);
        break;
    default:
        return -1;
        break;
    }
}

#define INSTANTIATE(T)                                                              \
    template int homography(Array<T> &H,                                            \
                            const Array<float> &x_src, const Array<float> &y_src,   \
                            const Array<float> &x_dst, const Array<float> &y_dst,   \
                            const af_homography_type htype, const float inlier_thr, \
                            const unsigned iterations);

INSTANTIATE(float )
INSTANTIATE(double)

}
