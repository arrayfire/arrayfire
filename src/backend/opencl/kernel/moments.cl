/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#define M00 moments_m00
#define M01 moments_m01
#define M10 moments_m10
#define M11 moments_m11

////////////////////////////////////////////////////////////////////////////////////
// Helper Functions
////////////////////////////////////////////////////////////////////////////////////
inline void fatomic_add_g(volatile __global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal, prevVal, expVal;

    prevVal.floatVal = *source;
    do {
        expVal.floatVal = prevVal.floatVal;
        newVal.floatVal = expVal.floatVal + operand;
        prevVal.intVal  = atomic_cmpxchg((volatile __global unsigned int *)source, expVal.intVal, newVal.intVal);
    } while (expVal.intVal != prevVal.intVal);
}

inline void fatomic_add_l(volatile __local float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal, prevVal, expVal;

    prevVal.floatVal = *source;
    do {
        expVal.floatVal = prevVal.floatVal;
        newVal.floatVal = expVal.floatVal + operand;
        prevVal.intVal  = atomic_cmpxchg((volatile __local unsigned int *)source, expVal.intVal, newVal.intVal);
    } while (expVal.intVal != prevVal.intVal);
}

///////////////////////////////////////////////////////////////////////////
// moments
///////////////////////////////////////////////////////////////////////////
float moments_m00(const int mId, const int idx, const int idy,
                   __global const T *d_in,  const KParam in) {
    return d_in[mId];
}

float moments_m01(const int mId, const int idx, const int idy,
                   __global const T *d_in,  const KParam in) {
    return idx * d_in[mId];
}

float moments_m10(const int mId, const int idx, const int idy,
                   __global const T *d_in,  const KParam in) {
    return idy * d_in[mId];
}

float moments_m11(const int mId, const int idx, const int idy,
                   __global const T *d_in,  const KParam in) {
    return idx * idy * d_in[mId];
}

////////////////////////////////////////////////////////////////////////////////////
// Wrapper Kernel
////////////////////////////////////////////////////////////////////////////////////
__kernel
void moments_kernel(__global  float *d_out, const KParam out,
                    __global const T *d_in,  const KParam in,
                    const int blocksMatX, const int pBatch)
{
    const int idw = get_group_id(1) / in.dims[2];
    const int idz = get_group_id(1)  - idw * in.dims[2];

    const int idy = get_group_id(0) / blocksMatX;
    const int blockIdx_x = get_group_id(0) - idy * blocksMatX;
    const int idx = get_local_id(0) + blockIdx_x * get_local_size(0);

    int mId = idy * in.strides[1] + idx;
    if(pBatch) {
       mId += idw * in.strides[3] + idz * in.strides[2];
    }

    if(idx >= in.dims[0] ||
       idy >= in.dims[1] ||
       idz >= in.dims[2] ||
       idw >= in.dims[3])
        return;

    __local float wkg_moment_sum;
    wkg_moment_sum = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    fatomic_add_l(&wkg_moment_sum, MOMENT(mId, idx, idy, d_in, in));
    barrier(CLK_LOCAL_MEM_FENCE);

    if(get_local_id(0) == 0)
        fatomic_add_g(d_out + (idw * out.strides[1] + idz), wkg_moment_sum);

}
