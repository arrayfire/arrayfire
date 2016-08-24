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
             const T beta,
             __global int *counter)
{

    int rowNext = get_global_id(0);
    while (true) {
#if USE_GREEDY
        int rowId = atomic_inc(counter);
#else
        int rowId = rowNext;
        rowNext += get_global_size(0);
#endif
        if (rowId >= M) return;

        int colStart = rowidx[rowId];
        int colEnd   = rowidx[rowId + 1];
        T outval = 0;

        for (int id = colStart; id < colEnd; id++) {
            int cid = colidx[id];
            outval += MUL(values[id], rhs[cid]);
        }

#if USE_ALPHA
        outval *= alpa;
#endif

#if USE_BETA
        output[rowId] = outval + beta * output[rowId];
#else
        output[rowId] = outval;
#endif
    }
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
            const T beta,
            __global int *counter)
{
    int lid = get_local_id(0);
    int off = get_local_size(0);

    int rowNext = get_group_id(0);
    __local int s_rowId;
    while (true) {

#if USE_GREEDY
        if (lid == 0) {
            s_rowId = atomic_inc(counter);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        int rowId = s_rowId;
#else
        int rowId = rowNext;
        rowNext += get_num_groups(0);
#endif
        if (rowId >= M) return;

        __local T s_outval[THREADS];

        int colStart = rowidx[rowId];
        int colEnd   = rowidx[rowId + 1];
        T outval = 0;
        for (int id = colStart + lid; id < colEnd; id += off) {
            int cid = colidx[id];
            outval += MUL(values[id], rhs[cid]);
        }
        s_outval[lid] = outval;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int n = THREADS / 2; n > 0; n /= 2) {
            if (lid < n) s_outval[lid] += s_outval[lid + n];
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
#if USE_ALPHA
            outval = alpha * s_outval[0];
#else
            outval = s_outval[0];
#endif

#if USE_BETA
            output[rowId] = outval + beta * output[rowId];
#else
            output[rowId] = outval;
#endif
        }
    }
}
