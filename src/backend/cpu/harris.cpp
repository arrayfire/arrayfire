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
#include <af/constants.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <err_cpu.hpp>
#include <handle.hpp>
#include <harris.hpp>
#include <convolve.hpp>
#include <gradient.hpp>
#include <sort_index.hpp>
#include <cstring>

using af::dim4;

namespace cpu
{

template<typename T>
void gaussian1D(T* out, const int dim, double sigma=0.0)
{
    if(!(sigma>0)) sigma = 0.25*dim;

    T sum = (T)0;
    for(int i=0;i<dim;i++)
    {
        int x = i-(dim-1)/2;
        T el = 1. / sqrt(2 * af::Pi * sigma*sigma) * exp(-((x*x)/(2*(sigma*sigma))));
        out[i] = el;
        sum   += el;
    }

    for(int k=0;k<dim;k++)
        out[k] /= sum;
}

template<typename T>
void second_order_deriv(
    T* ixx_out,
    T* ixy_out,
    T* iyy_out,
    const unsigned in_len,
    const T* ix_in,
    const T* iy_in)
{
    for (unsigned x = 0; x < in_len; x++) {
        ixx_out[x] = ix_in[x] * ix_in[x];
        ixy_out[x] = ix_in[x] * iy_in[x];
        iyy_out[x] = iy_in[x] * iy_in[x];
    }
}

template<typename T>
void harris_responses(
    T* resp_out,
    const unsigned idim0,
    const unsigned idim1,
    const T* ixx_in,
    const T* ixy_in,
    const T* iyy_in,
    const float k_thr,
    const unsigned border_len)
{
    const unsigned r = border_len;

    for (unsigned x = r; x < idim1 - r; x++) {
        for (unsigned y = r; y < idim0 - r; y++) {
            const unsigned idx = x * idim0 + y;

            // Calculates matrix trace and determinant
            T tr = ixx_in[idx] + iyy_in[idx];
            T det = ixx_in[idx] * iyy_in[idx] - ixy_in[idx] * ixy_in[idx];

            // Calculates local Harris response
            resp_out[idx] = det - k_thr * (tr*tr);
        }
    }
}

template<typename T>
void non_maximal(
    float* x_out,
    float* y_out,
    float* resp_out,
    unsigned* count,
    const unsigned idim0,
    const unsigned idim1,
    const T* resp_in,
    const float min_resp,
    const unsigned border_len,
    const unsigned max_corners)
{
    // Responses on the border don't have 8-neighbors to compare, discard them
    const unsigned r = border_len + 1;

    for (unsigned x = r; x < idim1 - r; x++) {
        for (unsigned y = r; y < idim0 - r; y++) {
            const T v = resp_in[x * idim0 + y];

            // Find maximum neighborhood response
            T max_v;
            max_v = max(resp_in[(x-1) * idim0 + y-1], resp_in[x * idim0 + y-1]);
            max_v = max(max_v, resp_in[(x+1) * idim0 + y-1]);
            max_v = max(max_v, resp_in[(x-1) * idim0 + y  ]);
            max_v = max(max_v, resp_in[(x+1) * idim0 + y  ]);
            max_v = max(max_v, resp_in[(x-1) * idim0 + y+1]);
            max_v = max(max_v, resp_in[(x)   * idim0 + y+1]);
            max_v = max(max_v, resp_in[(x+1) * idim0 + y+1]);

            // Stores corner to {x,y,resp}_out if it's response is maximum compared
            // to its 8-neighborhood and greater or equal minimum response
            if (v > max_v && v >= (T)min_resp) {
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

void keep_corners(
    float* x_out,
    float* y_out,
    float* resp_out,
    const float* x_in,
    const float* y_in,
    const float* resp_in,
    const unsigned* resp_idx,
    const unsigned n_corners)
{
    // Keep only the first n_feat features
    for (unsigned f = 0; f < n_corners; f++) {
        x_out[f] = x_in[resp_idx[f]];
        y_out[f] = y_in[resp_idx[f]];
        resp_out[f] = resp_in[f];
    }
}

template<typename T, typename convAccT>
unsigned harris(Array<float> &x_out, Array<float> &y_out, Array<float> &resp_out,
                const Array<T> &in, const unsigned max_corners, const float min_response,
                const float sigma, const unsigned filter_len, const float k_thr)
{
    dim4 idims = in.dims();

    // Window filter
    convAccT* h_filter = memAlloc<convAccT>(filter_len);
    // Decide between rectangular or circular filter
    if (sigma < 0.5f) {
        for (unsigned i = 0; i < filter_len; i++)
            h_filter[i] = (T)1.f / (filter_len);
    }
    else {
        gaussian1D<convAccT>(h_filter, (int)filter_len, sigma);
    }
    Array<convAccT> filter = createHostDataArray(filter_len, h_filter);

    unsigned border_len = filter_len / 2 + 1;

    Array<T> ix = createEmptyArray<T>(idims);
    Array<T> iy = createEmptyArray<T>(idims);

    // Compute first order derivatives
    gradient<T>(iy, ix, in);

    Array<T> ixx = createEmptyArray<T>(idims);
    Array<T> ixy = createEmptyArray<T>(idims);
    Array<T> iyy = createEmptyArray<T>(idims);

    // Compute second-order derivatives
    second_order_deriv<T>(ixx.get(), ixy.get(), iyy.get(),
                          in.elements(), ix.get(), iy.get());

    // Convolve second-order derivatives with proper window filter
    ixx = convolve2<T, convAccT, false>(ixx, filter, filter);
    ixy = convolve2<T, convAccT, false>(ixy, filter, filter);
    iyy = convolve2<T, convAccT, false>(iyy, filter, filter);

    const unsigned corner_lim = in.elements() * 0.2f;

    float* x_corners = memAlloc<float>(corner_lim);
    float* y_corners = memAlloc<float>(corner_lim);
    float* resp_corners = memAlloc<float>(corner_lim);

    T* resp = memAlloc<T>(in.elements());

    // Calculate Harris responses for all pixels
    harris_responses<T>(resp,
                        idims[0], idims[1],
                        ixx.get(), ixy.get(), iyy.get(),
                        k_thr, border_len);

    const unsigned min_r = (max_corners > 0) ? 0.f : min_response;
    unsigned corners_found = 0;

    // Performs non-maximal suppression
    non_maximal<T>(x_corners, y_corners, resp_corners, &corners_found,
                   idims[0], idims[1], resp, min_r, border_len, corner_lim);

    memFree(resp);

    const unsigned corners_out = (max_corners > 0) ?
                                 min(corners_found, max_corners) :
                                 min(corners_found, corner_lim);
    if (corners_out == 0)
        return 0;

    if (max_corners > 0 && corners_found > corners_out) {
        Array<float> harris_responses = createHostDataArray<float>(dim4(corners_found), resp_corners);
        Array<float> harris_sorted = createEmptyArray<float>(dim4(corners_found));
        Array<unsigned> harris_idx = createEmptyArray<unsigned>(dim4(corners_found));

        // Sort Harris responses
        sort_index<float, false>(harris_sorted, harris_idx, harris_responses, 0);

        x_out = createEmptyArray<float>(dim4(corners_out));
        y_out = createEmptyArray<float>(dim4(corners_out));
        resp_out = createEmptyArray<float>(dim4(corners_out));

        // Keep only the corners with higher Harris responses
        keep_corners(x_out.get(), y_out.get(), resp_out.get(),
                     x_corners, y_corners, harris_sorted.get(), harris_idx.get(),
                     corners_out);

        memFree(x_corners);
        memFree(y_corners);
    }
    else if (max_corners == 0 && corners_found < corner_lim) {
        x_out = createEmptyArray<float>(dim4(corners_out));
        y_out = createEmptyArray<float>(dim4(corners_out));
        resp_out = createEmptyArray<float>(dim4(corners_out));

        memcpy(x_out.get(), x_corners, corners_out * sizeof(float));
        memcpy(y_out.get(), y_corners, corners_out * sizeof(float));
        memcpy(resp_out.get(), resp_corners, corners_out * sizeof(float));

        memFree(x_corners);
        memFree(y_corners);
        memFree(resp_corners);
    }
    else {
        x_out = createHostDataArray<float>(dim4(corners_out), x_corners);
        y_out = createHostDataArray<float>(dim4(corners_out), y_corners);
        resp_out = createHostDataArray<float>(dim4(corners_out), resp_corners);
    }

    return corners_out;
}

#define INSTANTIATE(T, convAccT)                                                                                    \
    template unsigned harris<T, convAccT>(Array<float> &x_out, Array<float> &y_out, Array<float> &score_out,        \
                                          const Array<T> &in, const unsigned max_corners, const float min_response, \
                                          const float sigma, const unsigned block_size, const float k_thr);

INSTANTIATE(double, double)
INSTANTIATE(float , float)

}
