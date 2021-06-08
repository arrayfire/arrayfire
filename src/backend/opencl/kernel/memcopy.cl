/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

typedef struct {
    int dims[4];
} dims_t;

kernel void memCopy(global T *d_out, const dims_t ostrides, const int ooffset,
                    global const T *d_in, const dims_t idims,
                    const dims_t istrides, const int ioffset) {
    const int g0     = get_global_id(0);  // dim[0]
    const int g1     = get_global_id(1);  // dim[1]
    const bool valid = (g0 < (int)idims.dims[0]) && (g1 < (int)idims.dims[1]);
    if (valid) {
        const int g2 = get_global_id(2);  // dim[2] never overflow
                                          // dim[3] is through loop
        int idx_in = ioffset + g0 * (int)istrides.dims[0] +
                     g1 * (int)istrides.dims[1] + g2 * (int)istrides.dims[2];
        const int istrides3 = istrides.dims[3];
        const int idx_inEnd = idx_in + (int)idims.dims[3] * istrides3;
        int idx_out         = ooffset + g0 * (int)ostrides.dims[0] +
                      g1 * (int)ostrides.dims[1] + g2 * (int)ostrides.dims[2];
        const int ostrides3 = ostrides.dims[3];
        do {
            T val = d_in[idx_in];
            idx_in += istrides3;
            d_out[idx_out] = val;
            idx_out += ostrides3;
        } while (idx_in != idx_inEnd);
    }
}
