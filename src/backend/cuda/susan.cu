/*******************************************************
 * Copyright (c) 2015, Arrayfire
 * all rights reserved.
 *
 * This file is distributed under 3-clause bsd license.
 * the complete license agreement can be obtained at:
 * http://Arrayfire.com/licenses/bsd-3-clause
 ********************************************************/

#include <af/features.h>
#include <Array.hpp>
#include <err_cuda.hpp>
#include <susan.hpp>
#include <kernel/susan.hpp>

using af::features;

namespace cuda
{

template<typename T>
unsigned susan(Array<float> &x_out, Array<float> &y_out, Array<float> &resp_out,
               const Array<T> &in,
               const unsigned radius, const float diff_thr, const float geom_thr,
               const float feature_ratio, const unsigned edge)
{
    dim4 idims = in.dims();

    const unsigned corner_lim = in.elements() * feature_ratio;
    float* x_corners          = memAlloc<float>(corner_lim);
    float* y_corners          = memAlloc<float>(corner_lim);
    float* resp_corners       = memAlloc<float>(corner_lim);

    T* resp = memAlloc<T>(in.elements());
    unsigned corners_found = 0;

    kernel::susan_responses<T>(resp, in.get(), idims[0], idims[1], radius, diff_thr, geom_thr, edge);

    kernel::nonMaximal<T>(x_corners, y_corners, resp_corners, &corners_found,
                           idims[0], idims[1], resp, edge, corner_lim);

    memFree(resp);

    const unsigned corners_out = min(corners_found, corner_lim);
    if (corners_out == 0)
        return 0;

    x_out = createDeviceDataArray<float>(dim4(corners_out), (void*)x_corners);
    y_out = createDeviceDataArray<float>(dim4(corners_out), (void*)y_corners);
    resp_out = createDeviceDataArray<float>(dim4(corners_out), (void*)resp_corners);

    return corners_out;
}

#define INSTANTIATE(T) \
template unsigned susan<T>(Array<float> &x_out, Array<float> &y_out, Array<float> &score_out,   \
                           const Array<T> &in, const unsigned radius, const float diff_thr,     \
                           const float geom_thr, const float feature_ratio, const unsigned edge);

INSTANTIATE(float )
INSTANTIATE(double)
INSTANTIATE(char  )
INSTANTIATE(int   )
INSTANTIATE(uint  )
INSTANTIATE(uchar )

}
