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
using std::vector;

namespace arrayfire {
namespace opencl {

template<typename T>
unsigned susan(Array<float> &x_out, Array<float> &y_out, Array<float> &resp_out,
               const Array<T> &in, const unsigned radius, const float diff_thr,
               const float geom_thr, const float feature_ratio,
               const unsigned edge) {
    dim4 idims = in.dims();

    const unsigned corner_lim = in.elements() * feature_ratio;
    Array<float> x_corners    = createEmptyArray<float>({corner_lim});
    Array<float> y_corners    = createEmptyArray<float>({corner_lim});
    Array<float> resp_corners = createEmptyArray<float>({corner_lim});

    auto resp = memAlloc<float>(in.elements());

    kernel::susan<T>(resp.get(), in.get(), in.getOffset(), idims[0], idims[1],
                     diff_thr, geom_thr, edge, radius);

    unsigned corners_found = kernel::nonMaximal<T>(
        x_corners.get(), y_corners.get(), resp_corners.get(), idims[0],
        idims[1], resp.get(), edge, corner_lim);

    const unsigned corners_out = std::min(corners_found, corner_lim);
    if (corners_out == 0) {
        x_out    = createEmptyArray<float>(dim4());
        y_out    = createEmptyArray<float>(dim4());
        resp_out = createEmptyArray<float>(dim4());
    } else {
        vector<af_seq> idx{{0., static_cast<double>(corners_out - 1.0), 1.}};
        x_out    = createSubArray(x_corners, idx);
        y_out    = createSubArray(y_corners, idx);
        resp_out = createSubArray(resp_corners, idx);
    }
    return corners_out;
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
}  // namespace arrayfire
