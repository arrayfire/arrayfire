/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/defines.h>
#include <af/vision.h>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <ArrayInfo.hpp>
#include <homography.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static inline void homography(af_array &H, int &inliers,
                              const af_array x_src, const af_array y_src,
                              const af_array x_dst, const af_array y_dst,
                              const af_homography_type htype, const float inlier_thr,
                              const unsigned iterations)
{
    Array<T> bestH = createEmptyArray<T>(af::dim4(3, 3));

    inliers = homography<T>(bestH,
                            getArray<float>(x_src), getArray<float>(y_src),
                            getArray<float>(x_dst), getArray<float>(y_dst),
                            htype, inlier_thr, iterations);

    H = getHandle<T>(bestH);
}

af_err af_homography(af_array *H, int *inliers,
                     const af_array x_src, const af_array y_src,
                     const af_array x_dst, const af_array y_dst,
                     const af_homography_type htype, const float inlier_thr,
                     const unsigned iterations, const af_dtype otype)
{
    try {
        ArrayInfo xsinfo = getInfo(x_src);
        ArrayInfo ysinfo = getInfo(y_src);
        ArrayInfo xdinfo = getInfo(x_dst);
        ArrayInfo ydinfo = getInfo(y_dst);

        af::dim4 xsdims  = xsinfo.dims();
        af::dim4 ysdims  = ysinfo.dims();
        af::dim4 xddims  = xdinfo.dims();
        af::dim4 yddims  = ydinfo.dims();

        af_dtype xstype = xsinfo.getType();
        af_dtype ystype = ysinfo.getType();
        af_dtype xdtype = xdinfo.getType();
        af_dtype ydtype = ydinfo.getType();

        if (xstype != f32) { TYPE_ERROR(1, xstype); }
        if (ystype != f32) { TYPE_ERROR(2, ystype); }
        if (xdtype != f32) { TYPE_ERROR(3, xdtype); }
        if (ydtype != f32) { TYPE_ERROR(4, ydtype); }

        ARG_ASSERT(1, (xsdims[0] > 0));
        ARG_ASSERT(2, (ysdims[0] == xsdims[0]));
        ARG_ASSERT(3, (xddims[0] > 0));
        ARG_ASSERT(4, (yddims[0] == yddims[0]));

        ARG_ASSERT(5, (inlier_thr >= 0.1f));
        ARG_ASSERT(6, (iterations > 0));

        af_array outH;
        int outInl;

        switch(otype) {
            case f32: homography<float >(outH, outInl, x_src, y_src, x_dst, y_dst, htype, inlier_thr, iterations);  break;
            case f64: homography<double>(outH, outInl, x_src, y_src, x_dst, y_dst, htype, inlier_thr, iterations);  break;
            default:  TYPE_ERROR(1, otype);
        }
        std::swap(*H, outH);
        std::swap(*inliers, outInl);
    }
    CATCHALL;

    return AF_SUCCESS;
}
