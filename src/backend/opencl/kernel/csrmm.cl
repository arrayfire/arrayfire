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
csrmm_nt(__global T *output,
         __global const T *values,
         __global const int *rowidx,
         __global const int *colidx,
         const int M,
         const int N,
         const int K,
         __global const T *rhs,
         const KParam rinfo,
         const T alpha,
         const T beta,
         __global int *counter)
{
    int gidx = get_global_id(0);
    int lid = get_local_id(0);
    int off = get_local_size(0);

    rhs += gidx;
    output += gidx * M;

    bool within_K = (gidx < K);

    __local T s_values[THREADS_PER_GROUP];
    __local int s_colidx[THREADS_PER_GROUP];

    // FIXME: Implement better load balancing using atomic counter
    for (int rid = get_group_id(1); rid < M; rid += get_num_groups(1)) {
        barrier(CLK_LOCAL_MEM_FENCE);

        const int colStart = rowidx[rid];
        const int colEnd   = rowidx[rid + 1];

        T outval = 0;
        for (int id = colStart; id < colEnd; id += off) {
            int lim = min(colEnd - id, off);
            s_values[lid] = lid < lim ? values[id + lid] : 0;
            s_colidx[lid] = lid < lim ? colidx[id + lid] : -1;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (int idy = 0; within_K && idy < lim; idy++) {
                outval += MUL(s_values[idy], rhs[K * s_colidx[idy]]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (within_K) {
#if USE_ALPHA
            outval = alpha * outval;
#endif

#if USE_BETA
            output[rid] = outval + beta * output[rid];
#else
            output[rid] = outval;
#endif
        }
    }
}
