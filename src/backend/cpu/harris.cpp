/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <convolve.hpp>
#include <gradient.hpp>
#include <harris.hpp>
#include <kernel/harris.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <sort_index.hpp>
#include <af/dim4.hpp>
#include <cstring>

using af::dim4;

namespace arrayfire {
namespace cpu {

template<typename T, typename convAccT>
unsigned harris(Array<float> &x_out, Array<float> &y_out,
                Array<float> &resp_out, const Array<T> &in,
                const unsigned max_corners, const float min_response,
                const float sigma, const unsigned filter_len,
                const float k_thr) {
    dim4 idims = in.dims();

    // Window filter
    auto h_filter = memAlloc<convAccT>(filter_len);
    // Decide between rectangular or circular filter
    if (sigma < 0.5f) {
        for (unsigned i = 0; i < filter_len; i++) {
            h_filter[i] = static_cast<T>(1) / (filter_len);
        }
    } else {
        gaussian1D<convAccT>(h_filter.get(), static_cast<int>(filter_len),
                             sigma);
    }
    Array<convAccT> filter =
        createDeviceDataArray<convAccT>(dim4(filter_len), h_filter.release());
    unsigned border_len = filter_len / 2 + 1;

    Array<T> ix = createEmptyArray<T>(idims);
    Array<T> iy = createEmptyArray<T>(idims);

    // Compute first order derivatives
    gradient<T>(iy, ix, in);

    Array<T> ixx = createEmptyArray<T>(idims);
    Array<T> ixy = createEmptyArray<T>(idims);
    Array<T> iyy = createEmptyArray<T>(idims);

    // Compute second-order derivatives
    getQueue().enqueue(kernel::second_order_deriv<T>, ixx, ixy, iyy,
                       in.elements(), ix, iy);

    // Convolve second-order derivatives with proper window filter
    ixx = convolve2<T, convAccT>(ixx, filter, filter, false);
    ixy = convolve2<T, convAccT>(ixy, filter, filter, false);
    iyy = convolve2<T, convAccT>(iyy, filter, filter, false);

    const unsigned corner_lim = in.elements() * 0.2f;

    Array<T> responses = createEmptyArray<T>(dim4(in.elements()));

    getQueue().enqueue(kernel::harris_responses<T>, responses, idims[0],
                       idims[1], ixx, ixy, iyy, k_thr, border_len);

    Array<float> xCorners    = createEmptyArray<float>(dim4(corner_lim));
    Array<float> yCorners    = createEmptyArray<float>(dim4(corner_lim));
    Array<float> respCorners = createEmptyArray<float>(dim4(corner_lim));

    const unsigned min_r =
        (max_corners > 0) ? 0U : static_cast<unsigned>(min_response);

    // Performs non-maximal suppression
    getQueue().sync();
    unsigned corners_found = 0;
    kernel::non_maximal<T>(xCorners, yCorners, respCorners, &corners_found,
                           idims[0], idims[1], responses, min_r, border_len,
                           corner_lim);

    const unsigned corners_out =
        min(corners_found, (max_corners > 0) ? max_corners : corner_lim);
    if (corners_out == 0) { return 0; }

    if (max_corners > 0 && corners_found > corners_out) {
        respCorners.resetDims(dim4(corners_found));
        Array<float> harris_sorted =
            createEmptyArray<float>(dim4(corners_found));
        Array<unsigned> harris_idx =
            createEmptyArray<unsigned>(dim4(corners_found));

        // Sort Harris responses
        sort_index<float>(harris_sorted, harris_idx, respCorners, 0, false);

        x_out    = createEmptyArray<float>(dim4(corners_out));
        y_out    = createEmptyArray<float>(dim4(corners_out));
        resp_out = createEmptyArray<float>(dim4(corners_out));

        // Keep only the corners with higher Harris responses
        getQueue().enqueue(kernel::keep_corners, x_out, y_out, resp_out,
                           xCorners, yCorners, harris_sorted, harris_idx,
                           corners_out);
    } else if (max_corners == 0 && corners_found < corner_lim) {
        x_out    = createEmptyArray<float>(dim4(corners_out));
        y_out    = createEmptyArray<float>(dim4(corners_out));
        resp_out = createEmptyArray<float>(dim4(corners_out));

        auto copyFunc =
            [=](Param<float> x_out, Param<float> y_out,
                Param<float> outResponses, const CParam<float> &x_crnrs,
                const CParam<float> &y_crnrs, const CParam<float> &inResponses,
                const unsigned corners_out) {
                memcpy(x_out.get(), x_crnrs.get(), corners_out * sizeof(float));
                memcpy(y_out.get(), y_crnrs.get(), corners_out * sizeof(float));
                memcpy(outResponses.get(), inResponses.get(),
                       corners_out * sizeof(float));
            };
        getQueue().enqueue(copyFunc, x_out, y_out, resp_out, xCorners, yCorners,
                           respCorners, corners_out);
    } else {
        x_out    = xCorners;
        y_out    = yCorners;
        resp_out = respCorners;
        x_out.resetDims(dim4(corners_out));
        y_out.resetDims(dim4(corners_out));
        resp_out.resetDims(dim4(corners_out));
    }

    return corners_out;
}

#define INSTANTIATE(T, convAccT)                                              \
    template unsigned harris<T, convAccT>(                                    \
        Array<float> & x_out, Array<float> & y_out, Array<float> & score_out, \
        const Array<T> &in, const unsigned max_corners,                       \
        const float min_response, const float sigma,                          \
        const unsigned block_size, const float k_thr);

INSTANTIATE(double, double)
INSTANTIATE(float, float)

}  // namespace cpu
}  // namespace arrayfire
