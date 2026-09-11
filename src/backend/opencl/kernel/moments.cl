/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#define AF_MOMENT_M00 1
#define AF_MOMENT_M01 2
#define AF_MOMENT_M10 4
#define AF_MOMENT_M11 8

inline void fatomic_add_g(volatile global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal, prevVal, expVal;

    prevVal.floatVal = *source;
    do {
        expVal.floatVal = prevVal.floatVal;
        newVal.floatVal = expVal.floatVal + operand;
        prevVal.intVal  = atomic_cmpxchg((volatile global unsigned int *)source,
                                        expVal.intVal, newVal.intVal);
    } while (expVal.intVal != prevVal.intVal);
}

kernel void moments(global float *d_out, const KParam out, global const T *d_in,
                    const KParam in, const int moment, const int pBatch) {
    const dim_t idw = get_group_id(1) / in.dims[2];
    const dim_t idz = get_group_id(1) - idw * in.dims[2];
    const dim_t idy = get_group_id(0);
    const int lid   = get_local_id(0);
    const int lsz   = get_local_size(0);
    // One slot per work-item; THREADS is the launch-site local size.
    local float wkg_moment_sum[MOMENTS_SZ][THREADS];

    // Every work-item in the group shares idy/idz/idw, so the whole group
    // leaves together and the barriers below stay balanced.
    if (idy >= in.dims[1] || idz >= in.dims[2] || idw >= in.dims[3]) return;

    // Each work-item accumulates its rows privately; the group then reduces
    // in local memory with a tree. No local-memory atomics: the float
    // compare-and-swap loop this replaced returned zeros for every moment
    // slot past the first on Intel GPUs.
    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    dim_t mId = idy * in.strides[1] + lid;
    if (pBatch) { mId += idw * in.strides[3] + idz * in.strides[2]; }

    const float fy = (float)idy;
    for (dim_t idx = lid; idx < in.dims[0]; idx += lsz, mId += lsz) {
        const float val = d_in[mId];
        const float fx  = (float)idx;
        int m           = 0;
        if ((moment & AF_MOMENT_M00) > 0) { acc[m++] += val; }
        if ((moment & AF_MOMENT_M01) > 0) { acc[m++] += fx * val; }
        if ((moment & AF_MOMENT_M10) > 0) { acc[m++] += fy * val; }
        if ((moment & AF_MOMENT_M11) > 0) { acc[m++] += fx * fy * val; }
    }

    for (int m = 0; m < MOMENTS_SZ; ++m) { wkg_moment_sum[m][lid] = acc[m]; }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsz / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            for (int m = 0; m < MOMENTS_SZ; ++m) {
                wkg_moment_sum[m][lid] += wkg_moment_sum[m][lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid < MOMENTS_SZ) {
        fatomic_add_g(
            d_out + (idw * out.strides[3] + idz * out.strides[2]) + lid,
            wkg_moment_sum[lid][0]);
    }
}
