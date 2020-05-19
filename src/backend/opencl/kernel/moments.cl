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

inline void fatomic_add_l(volatile local float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal, prevVal, expVal;

    prevVal.floatVal = *source;
    do {
        expVal.floatVal = prevVal.floatVal;
        newVal.floatVal = expVal.floatVal + operand;
        prevVal.intVal  = atomic_cmpxchg((volatile local unsigned int *)source,
                                        expVal.intVal, newVal.intVal);
    } while (expVal.intVal != prevVal.intVal);
}

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
    dim_t idx       = get_local_id(0);

    if (idy >= in.dims[1] || idz >= in.dims[2] || idw >= in.dims[3]) return;

    local float wkg_moment_sum[MOMENTS_SZ];
    if (get_local_id(0) < MOMENTS_SZ) { wkg_moment_sum[get_local_id(0)] = 0.f; }
    barrier(CLK_LOCAL_MEM_FENCE);

    int mId = idy * in.strides[1] + idx;
    if (pBatch) { mId += idw * in.strides[3] + idz * in.strides[2]; }

    for (; idx < in.dims[0]; idx += get_local_size(0)) {
        dim_t m_off = 0;
        float val   = d_in[mId];
        mId += get_local_size(0);

        if ((moment & AF_MOMENT_M00) > 0) {
            fatomic_add_l(wkg_moment_sum + m_off++, val);
        }
        if ((moment & AF_MOMENT_M01) > 0) {
            fatomic_add_l(wkg_moment_sum + m_off++, idx * val);
        }
        if ((moment & AF_MOMENT_M10) > 0) {
            fatomic_add_l(wkg_moment_sum + m_off++, idy * val);
        }
        if ((moment & AF_MOMENT_M11) > 0) {
            fatomic_add_l(wkg_moment_sum + m_off, idx * idy * val);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(0) < out.dims[0])
        fatomic_add_g(d_out + (idw * out.strides[3] + idz * out.strides[2]) +
                          get_local_id(0),
                      wkg_moment_sum[get_local_id(0)]);
}
