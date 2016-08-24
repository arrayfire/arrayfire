/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if IS_CPLX

T __cmul(T lhs, T rhs)
{
    T out;
    out.x = lhs.x * rhs.x - lhs.y * rhs.y;
    out.y = lhs.x * rhs.y + lhs.y * rhs.x;
    return out;
}
#define MUL(a, b) __cmul(a, b)
#else
#define MUL(a, b) (a) * (b)
#endif


__kernel void
csrmv_thread(__global T *output,
             __global const T *values,
             __global const int *rowidx,
             __global const int *colidx,
             const int M,
             __global const T *rhs,
             const KParam rinfo,
             const T alpha,
             const T beta)
{
    int rid = get_global_id(0) + get_global_size(0) * get_global_id(1);
    if (rid >= M) return;

    int colStart = rowidx[rid];
    int colEnd   = rowidx[rid + 1];
    T val = 0;

    for (int id = colStart; id < colEnd; id++) {
        int cid = colidx[id];
        val += MUL(values[id], rhs[cid]);
    }

#if USE_ALPHA
    val *= alpa;
#endif

#if USE_BETA
    output[rid] = val + beta * output[rid];
#else
    output[rid] = val;
#endif
}

__kernel void
csrmv_block(__global T *output,
            __global const T *values,
            __global const int *rowidx,
            __global const int *colidx,
            const int M,
            __global const T *rhs,
            const KParam rinfo,
            const T alpha,
            const T beta)
{
    int rid = get_group_id(0) + get_num_groups(0) * get_group_id(1);
    int lid = get_local_id(0);
    int off = get_local_size(0);
    if (rid >= M) return;

    __local T s_val[THREADS_PER_GROUP];

    int colStart = rowidx[rid];
    int colEnd   = rowidx[rid + 1];
    T val = 0;
    for (int id = colStart + lid; id < colEnd; id += off) {
        int cid = colidx[id];
        val += MUL(values[id], rhs[cid]);
    }
    s_val[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int n = off / 2; n > 0; n /= 2) {
        if (lid < n) s_val[lid] += s_val[lid + n];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
#if USE_ALPHA
        val = alpha * s_val[0];
#else
        val = s_val[0];
#endif

#if USE_BETA
        output[rid] = val + beta * output[rid];
#else
        output[rid] = val;
#endif
    }
}
