/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <homography.hpp>

#include <arith.hpp>
#include <kernel/homography.hpp>
#include <af/dim4.hpp>

#include <algorithm>
#include <limits>

using af::dim4;
using std::numeric_limits;

namespace arrayfire {
namespace opencl {

template<typename T>
int homography(Array<T> &bestH, const Array<float> &x_src,
               const Array<float> &y_src, const Array<float> &x_dst,
               const Array<float> &y_dst, const Array<float> &initial,
               const af_homography_type htype, const float inlier_thr,
               const unsigned iterations) {
    // constexpr float RANSACConfidence  = 0.99f;
    constexpr float LMEDSConfidence   = 0.99f;
    constexpr float LMEDSOutlierRatio = 0.4f;

    const af::dim4 &idims   = x_src.dims();
    const unsigned nsamples = idims[0];

    unsigned iter    = iterations;
    Array<float> err = createEmptyArray<float>(af::dim4());
    if (htype == AF_HOMOGRAPHY_LMEDS) {
        iter =
            ::std::min(iter, static_cast<unsigned>(
                                 log(1.f - LMEDSConfidence) /
                                 log(1.f - pow(1.f - LMEDSOutlierRatio, 4.f))));
        err = createValueArray<float>(af::dim4(nsamples, iter),
                                      numeric_limits<float>::max());
    } else {
        // Avoid passing "null" cl_mem object to kernels
        err = createEmptyArray<float>(af::dim4(1));
    }

    const size_t iter_sz = divup(iter, 256) * 256;

    af::dim4 rdims(4, iter_sz);
    Array<float> fctr =
        createValueArray<float>(rdims, static_cast<float>(nsamples));
    Array<float> rnd = arithOp<float, af_mul_t>(initial, fctr, rdims);

    Array<T> tmpH =
        createValueArray<T>(af::dim4(9, iter_sz), static_cast<T>(0));

    bestH = createValueArray<T>(af::dim4(3, 3), static_cast<T>(0));
    return kernel::computeH<T>(bestH, tmpH, err, x_src, y_src, x_dst, y_dst,
                               rnd, iter, nsamples, inlier_thr, htype);
}

#define INSTANTIATE(T)                                                     \
    template int homography(                                               \
        Array<T> &H, const Array<float> &x_src, const Array<float> &y_src, \
        const Array<float> &x_dst, const Array<float> &y_dst,              \
        const Array<float> &initial, const af_homography_type htype,       \
        const float inlier_thr, const unsigned iterations);

INSTANTIATE(float)
INSTANTIATE(double)

}  // namespace opencl
}  // namespace arrayfire
