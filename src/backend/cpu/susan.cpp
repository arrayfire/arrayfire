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
#include <cmath>
#include <math.hpp>

using af::features;

namespace cpu
{

template<typename T>
void susan_responses(T* resp_out, const T* in,
                     const unsigned idim0, const unsigned idim1,
                     const int radius, const float t, const float g,
                     const unsigned border_len)
{
    const unsigned r = border_len;
    const int rSqrd = radius*radius;

    for (unsigned y = r; y < idim1 - r; ++y) {
        for (unsigned x = r; x < idim0 - r; ++x) {
            const unsigned idx = y * idim0 + x;
            T m_0 = in[idx];
            float nM = 0.0f;

            for (int i=-radius; i<=radius; ++i) {
                for (int j=-radius; j<=radius; ++j) {
                    if (i*i + j*j < rSqrd) {
                        int p = x + i;
                        int q = y + j;
                        T m = in[p + idim0 * q];
                        float exp_pow = std::pow((m - m_0)/t, 6.0);
                        float cM = std::exp(-exp_pow);
                        nM += cM;
                    }
                }
            }

            resp_out[idx] = nM < g ? g - nM : T(0);
        }
    }
}

template<typename T>
void non_maximal(float* x_out, float* y_out, float* resp_out,
                 unsigned* count, const unsigned idim0, const unsigned idim1,
                 const T* resp_in, const unsigned border_len, const unsigned max_corners)
{
    // Responses on the border don't have 8-neighbors to compare, discard them
    const unsigned r = border_len + 1;

    for (unsigned y = r; y < idim1 - r; y++) {
        for (unsigned x = r; x < idim0 - r; x++) {
            const T v = resp_in[y * idim0 + x];

            // Find maximum neighborhood response
            T max_v;
            max_v = max(resp_in[(y-1) * idim0 + x-1], resp_in[y * idim0 + x-1]);
            max_v = max(max_v, resp_in[(y+1) * idim0 + x-1]);
            max_v = max(max_v, resp_in[(y-1) * idim0 + x  ]);
            max_v = max(max_v, resp_in[(y+1) * idim0 + x  ]);
            max_v = max(max_v, resp_in[(y-1) * idim0 + x+1]);
            max_v = max(max_v, resp_in[(y)   * idim0 + x+1]);
            max_v = max(max_v, resp_in[(y+1) * idim0 + x+1]);

            // Stores corner to {x,y,resp}_out if it's response is maximum compared
            // to its 8-neighborhood and greater or equal minimum response
            if (v > max_v) {
                const unsigned idx = *count;
                *count += 1;
                if (idx < max_corners) {
                    x_out[idx]    = (float)x;
                    y_out[idx]    = (float)y;
                    resp_out[idx] = (float)v;
                }
            }
        }
    }
}

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

    susan_responses<T>(resp, in.get(), idims[0], idims[1], radius, diff_thr, geom_thr, edge);

    non_maximal<T>(x_corners, y_corners, resp_corners, &corners_found,
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
