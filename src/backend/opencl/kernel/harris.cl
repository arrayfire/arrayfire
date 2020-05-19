/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#define MAX_VAL(A, B) (A) < (B) ? (B) : (A)

kernel void second_order_deriv(global T* ixx_out, global T* ixy_out,
                               global T* iyy_out, const dim_t in_len,
                               global const T* ix_in, global const T* iy_in) {
    const unsigned x = get_global_id(0);

    if (x < in_len) {
        ixx_out[x] = ix_in[x] * ix_in[x];
        ixy_out[x] = ix_in[x] * iy_in[x];
        iyy_out[x] = iy_in[x] * iy_in[x];
    }
}

kernel void harris_responses(global T* resp_out, const unsigned idim0,
                             const unsigned idim1, global const T* ixx_in,
                             global const T* ixy_in, global const T* iyy_in,
                             const float k_thr, const unsigned border_len) {
    const unsigned r = border_len;

    const unsigned x = get_global_id(0) + r;
    const unsigned y = get_global_id(1) + r;

    if (x < idim1 - r && y < idim0 - r) {
        const unsigned idx = x * idim0 + y;

        // Calculates matrix trace and determinant
        T tr  = ixx_in[idx] + iyy_in[idx];
        T det = ixx_in[idx] * iyy_in[idx] - ixy_in[idx] * ixy_in[idx];

        // Calculates local Harris response
        resp_out[idx] = det - k_thr * (tr * tr);
    }
}

kernel void non_maximal(global float* x_out, global float* y_out,
                        global float* resp_out, global unsigned* count,
                        global const T* resp_in, const unsigned idim0,
                        const unsigned idim1, const float min_resp,
                        const unsigned border_len, const unsigned max_corners) {
    // Responses on the border don't have 8-neighbors to compare, discard them
    const unsigned r = border_len + 1;

    const unsigned x = get_global_id(0) + r;
    const unsigned y = get_global_id(1) + r;

    if (x < idim1 - r && y < idim0 - r) {
        const T v = resp_in[x * idim0 + y];

        // Find maximum neighborhood response
        T max_v;
        max_v = MAX_VAL(resp_in[(x - 1) * idim0 + y - 1],
                        resp_in[x * idim0 + y - 1]);
        max_v = MAX_VAL(max_v, resp_in[(x + 1) * idim0 + y - 1]);
        max_v = MAX_VAL(max_v, resp_in[(x - 1) * idim0 + y]);
        max_v = MAX_VAL(max_v, resp_in[(x + 1) * idim0 + y]);
        max_v = MAX_VAL(max_v, resp_in[(x - 1) * idim0 + y + 1]);
        max_v = MAX_VAL(max_v, resp_in[(x)*idim0 + y + 1]);
        max_v = MAX_VAL(max_v, resp_in[(x + 1) * idim0 + y + 1]);

        // Stores corner to {x,y,resp}_out if it's response is maximum compared
        // to its 8-neighborhood and greater or equal minimum response
        if (v > max_v && v >= min_resp) {
            const unsigned idx = atomic_inc(count);
            if (idx < max_corners) {
                x_out[idx]    = (float)x;
                y_out[idx]    = (float)y;
                resp_out[idx] = (float)v;
            }
        }
    }
}

kernel void keep_corners(global float* x_out, global float* y_out,
                         global float* score_out, global const float* x_in,
                         global const float* y_in, global const float* score_in,
                         global const unsigned* score_idx,
                         const unsigned n_feat) {
    unsigned f = get_global_id(0);

    if (f < n_feat) {
        x_out[f]     = x_in[score_idx[f]];
        y_out[f]     = y_in[score_idx[f]];
        score_out[f] = score_in[f];
    }
}
