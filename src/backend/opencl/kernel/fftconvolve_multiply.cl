/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

kernel void complex_multiply(global CONVT *d_out, KParam oInfo,
                             global const CONVT *d_in1, KParam i1Info,
                             global const CONVT *d_in2, KParam i2Info,
                             const int nelem, const int kind) {
    const int t = get_global_id(0);

    if (t >= nelem) return;

    if (kind == AF_BATCH_NONE || kind == AF_BATCH_SAME) {
        // Complex multiply each signal to equivalent filter
        const int ridx = t * 2;
        const int iidx = t * 2 + 1;

        CONVT a = d_in1[i1Info.offset + ridx];
        CONVT b = d_in1[i1Info.offset + iidx];
        CONVT c = d_in2[i2Info.offset + ridx];
        CONVT d = d_in2[i2Info.offset + iidx];

        d_out[oInfo.offset + ridx] = a * c - b * d;
        d_out[oInfo.offset + iidx] = a * d + b * c;
    } else if (kind == AF_BATCH_LHS) {
        // Complex multiply all signals to filter
        const int ridx1 = t * 2;
        const int iidx1 = t * 2 + 1;

        // Treating complex output array as real-only array,
        // thus, multiply strides by 2
        const int ridx2 = ridx1 % (i2Info.strides[3] * i2Info.dims[3] * 2);
        const int iidx2 = iidx1 % (i2Info.strides[3] * i2Info.dims[3] * 2);

        CONVT a = d_in1[i1Info.offset + ridx1];
        CONVT b = d_in1[i1Info.offset + iidx1];
        CONVT c = d_in2[i2Info.offset + ridx2];
        CONVT d = d_in2[i2Info.offset + iidx2];

        d_out[oInfo.offset + ridx1] = a * c - b * d;
        d_out[oInfo.offset + iidx1] = a * d + b * c;
    } else if (kind == AF_BATCH_RHS) {
        // Complex multiply signal to all filters
        const int ridx2 = t * 2;
        const int iidx2 = t * 2 + 1;

        // Treating complex output array as real-only array,
        // thus, multiply strides by 2
        const int ridx1 = ridx2 % (i1Info.strides[3] * i1Info.dims[3] * 2);
        const int iidx1 = iidx2 % (i1Info.strides[3] * i1Info.dims[3] * 2);

        CONVT a = d_in1[i1Info.offset + ridx1];
        CONVT b = d_in1[i1Info.offset + iidx1];
        CONVT c = d_in2[i2Info.offset + ridx2];
        CONVT d = d_in2[i2Info.offset + iidx2];

        d_out[oInfo.offset + ridx2] = a * c - b * d;
        d_out[oInfo.offset + iidx2] = a * d + b * c;
    }
}
