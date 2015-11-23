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
#include <memory>
#include <platform.hpp>
#include <async_queue.hpp>

using af::features;
using std::shared_ptr;

namespace cpu
{

template<typename T>
void susan_responses(Array<T> output, const Array<T> input,
                     const unsigned idim0, const unsigned idim1,
                     const int radius, const float t, const float g,
                     const unsigned border_len)
{
    T* resp_out = output.get();
    const T* in = input.get();

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
void non_maximal(Array<float> xcoords, Array<float> ycoords, Array<float> response,
                 shared_ptr<unsigned> counter, const unsigned idim0, const unsigned idim1,
                 const Array<T> input, const unsigned border_len, const unsigned max_corners)
{
    float* x_out    = xcoords.get();
    float* y_out    = ycoords.get();
    float* resp_out = response.get();
    unsigned* count = counter.get();
    const T* resp_in= input.get();

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

    auto x_corners    = createEmptyArray<float>(dim4(corner_lim));
    auto y_corners    = createEmptyArray<float>(dim4(corner_lim));
    auto resp_corners = createEmptyArray<float>(dim4(corner_lim));
    auto response     = createEmptyArray<T>(dim4(in.elements()));
    auto corners_found= std::shared_ptr<unsigned>(memAlloc<unsigned>(1), memFree<unsigned>);
    corners_found.get()[0] = 0;

    getQueue().enqueue(susan_responses<T>, response, in, idims[0], idims[1],
                       radius, diff_thr, geom_thr, edge);
    getQueue().enqueue(non_maximal<T>, x_corners, y_corners, resp_corners, corners_found,
                       idims[0], idims[1], response, edge, corner_lim);
    getQueue().sync();

    const unsigned corners_out = min((corners_found.get())[0], corner_lim);
    if (corners_out == 0) {
        x_out    = createEmptyArray<float>(dim4());
        y_out    = createEmptyArray<float>(dim4());
        resp_out = createEmptyArray<float>(dim4());
        return 0;
    } else {
        x_out = x_corners;
        y_out = y_corners;
        resp_out = resp_corners;
        x_out.resetDims(dim4(corners_out));
        y_out.resetDims(dim4(corners_out));
        resp_out.resetDims(dim4(corners_out));
        return corners_out;
    }
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
INSTANTIATE(short)
INSTANTIATE(ushort)

}
