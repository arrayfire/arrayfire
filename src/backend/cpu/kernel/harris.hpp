/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <utility.hpp>

namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename T>
void second_order_deriv(Param<T> ixx, Param<T> ixy, Param<T> iyy,
                        const unsigned in_len, CParam<T> ix, CParam<T> iy) {
    T* ixx_out     = ixx.get();
    T* ixy_out     = ixy.get();
    T* iyy_out     = iyy.get();
    const T* ix_in = ix.get();
    const T* iy_in = iy.get();
    for (unsigned x = 0; x < in_len; x++) {
        ixx_out[x] = ix_in[x] * ix_in[x];
        ixy_out[x] = ix_in[x] * iy_in[x];
        iyy_out[x] = iy_in[x] * iy_in[x];
    }
}

template<typename T>
void harris_responses(Param<T> resp, const unsigned idim0, const unsigned idim1,
                      CParam<T> ixx, CParam<T> ixy, CParam<T> iyy,
                      const float k_thr, const unsigned border_len) {
    T* resp_out      = resp.get();
    const T* ixx_in  = ixx.get();
    const T* ixy_in  = ixy.get();
    const T* iyy_in  = iyy.get();
    const unsigned r = border_len;

    for (unsigned x = r; x < idim1 - r; x++) {
        for (unsigned y = r; y < idim0 - r; y++) {
            const unsigned idx = x * idim0 + y;

            // Calculates matrix trace and determinant
            T tr  = ixx_in[idx] + iyy_in[idx];
            T det = ixx_in[idx] * iyy_in[idx] - ixy_in[idx] * ixy_in[idx];

            // Calculates local Harris response
            resp_out[idx] = det - k_thr * (tr * tr);
        }
    }
}

template<typename T>
void non_maximal(Param<float> xOut, Param<float> yOut, Param<float> respOut,
                 unsigned* count, const unsigned idim0, const unsigned idim1,
                 CParam<T> respIn, const float min_resp,
                 const unsigned border_len, const unsigned max_corners) {
    float* x_out     = xOut.get();
    float* y_out     = yOut.get();
    float* resp_out  = respOut.get();
    const T* resp_in = respIn.get();
    // Responses on the border don't have 8-neighbors to compare, discard them
    const unsigned r = border_len + 1;

    for (unsigned x = r; x < idim1 - r; x++) {
        for (unsigned y = r; y < idim0 - r; y++) {
            const T v = resp_in[x * idim0 + y];

            // Find maximum neighborhood response
            T max_v;
            max_v = std::max(resp_in[(x - 1) * idim0 + y - 1],
                             resp_in[x * idim0 + y - 1]);
            max_v = std::max(max_v, resp_in[(x + 1) * idim0 + y - 1]);
            max_v = std::max(max_v, resp_in[(x - 1) * idim0 + y]);
            max_v = std::max(max_v, resp_in[(x + 1) * idim0 + y]);
            max_v = std::max(max_v, resp_in[(x - 1) * idim0 + y + 1]);
            max_v = std::max(max_v, resp_in[(x)*idim0 + y + 1]);
            max_v = std::max(max_v, resp_in[(x + 1) * idim0 + y + 1]);

            // Stores corner to {x,y,resp}_out if it's response is maximum
            // compared to its 8-neighborhood and greater or equal minimum
            // response
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

static void keep_corners(Param<float> xOut, Param<float> yOut,
                         Param<float> respOut, CParam<float> xIn,
                         CParam<float> yIn, CParam<float> respIn,
                         CParam<unsigned> respIdx, const unsigned n_corners) {
    float* x_out         = xOut.get();
    float* y_out         = yOut.get();
    float* resp_out      = respOut.get();
    const float* x_in    = xIn.get();
    const float* y_in    = yIn.get();
    const float* resp_in = respIn.get();
    const uint* resp_idx = respIdx.get();

    // Keep only the first n_feat features
    for (unsigned f = 0; f < n_corners; f++) {
        x_out[f]    = x_in[resp_idx[f]];
        y_out[f]    = y_in[resp_idx[f]];
        resp_out[f] = resp_in[f];
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
