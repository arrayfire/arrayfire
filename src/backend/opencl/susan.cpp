/*******************************************************
 * Copyright (c) 2015, Arrayfire
 * all rights reserved.
 *
 * This file is distributed under 3-clause bsd license.
 * the complete license agreement can be obtained at:
 * http://Arrayfire.com/licenses/bsd-3-clause
 ********************************************************/

#include <Array.hpp>
#include <err_opencl.hpp>
#include <kernel/susan.hpp>
#include <af/features.h>
#include <algorithm>
#include <cmath>

using af::features;

namespace opencl {

template<typename T>
unsigned susan(Array<float> &x_out, Array<float> &y_out, Array<float> &resp_out,
               const Array<T> &in, const unsigned radius, const float diff_thr,
               const float geom_thr, const float feature_ratio,
               const unsigned edge) {
    dim4 idims = in.dims();

    const unsigned corner_lim = in.elements() * feature_ratio;
    cl::Buffer *x_corners     = bufferAlloc(corner_lim * sizeof(float));
    cl::Buffer *y_corners     = bufferAlloc(corner_lim * sizeof(float));
    cl::Buffer *resp_corners  = bufferAlloc(corner_lim * sizeof(float));

    cl::Buffer *resp = bufferAlloc(in.elements() * sizeof(float));

    kernel::susan<T>(resp, in.get(), in.getOffset(), idims[0], idims[1],
                     diff_thr, geom_thr, edge, radius);

    unsigned corners_found =
        kernel::nonMaximal<T>(x_corners, y_corners, resp_corners, idims[0],
                              idims[1], resp, edge, corner_lim);
    bufferFree(resp);

    const unsigned corners_out = std::min(corners_found, corner_lim);
    if (corners_out == 0) {
        bufferFree(x_corners);
        bufferFree(y_corners);
        bufferFree(resp_corners);
        x_out    = createEmptyArray<float>(dim4());
        y_out    = createEmptyArray<float>(dim4());
        resp_out = createEmptyArray<float>(dim4());
        return 0;
    } else {
        x_out    = createDeviceDataArray<float>(dim4(corners_out),
                                             (void *)((*x_corners)()));
        y_out    = createDeviceDataArray<float>(dim4(corners_out),
                                             (void *)((*y_corners)()));
        resp_out = createDeviceDataArray<float>(dim4(corners_out),
                                                (void *)((*resp_corners)()));
        return corners_out;
    }
}

#define INSTANTIATE(T)                                                        \
    template unsigned susan<T>(                                               \
        Array<float> & x_out, Array<float> & y_out, Array<float> & score_out, \
        const Array<T> &in, const unsigned radius, const float diff_thr,      \
        const float geom_thr, const float feature_ratio, const unsigned edge);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(char)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)

}  // namespace opencl
