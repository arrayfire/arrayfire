/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

kernel void histogram(global uint *d_dst, KParam oInfo, global const T *d_src,
                      KParam iInfo, local uint *localMem, int len, int nbins,
                      float minval, float maxval, int nBBS) {
    unsigned b2 = get_group_id(0) / nBBS;
    int start = (get_group_id(0) - b2 * nBBS) * THRD_LOAD * get_local_size(0) +
                get_local_id(0);
    int end = min((int)(start + THRD_LOAD * get_local_size(0)), len);

    // offset input and output to account for batch ops
    global const T *in = d_src + b2 * iInfo.strides[2] +
                         get_group_id(1) * iInfo.strides[3] + iInfo.offset;
    global uint *out =
        d_dst + b2 * oInfo.strides[2] + get_group_id(1) * oInfo.strides[3];

    float dx = (maxval - minval) / (float)nbins;

    bool use_global = nbins > MAX_BINS;

    if (!use_global) {
        for (int i = get_local_id(0); i < nbins; i += get_local_size(0))
            localMem[i] = 0;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int row = start; row < end; row += get_local_size(0)) {
#if defined(IS_LINEAR)
        int idx = row;
#else
        int i0  = row % iInfo.dims[0];
        int i1  = row / iInfo.dims[0];
        int idx = i0 + i1 * iInfo.strides[1];
#endif
        int bin = (int)(((float)in[idx] - minval) / dx);
        bin     = max(bin, 0);
        bin     = min(bin, (int)nbins - 1);

        if (use_global) {
            atomic_inc((out + bin));
        } else {
            atomic_inc((localMem + bin));
        }
    }

    if (!use_global) {
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = get_local_id(0); i < nbins; i += get_local_size(0)) {
            atomic_add((out + i), localMem[i]);
        }
    }
}
