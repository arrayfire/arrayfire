/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if IS_CPLX
T __cmul(T lhs, T rhs) {
    T out;
    out.x = lhs.x * rhs.x - lhs.y * rhs.y;
    out.y = lhs.x * rhs.y + lhs.y * rhs.x;
    return out;
}

T __ccmul(T lhs, T rhs) {
    T out;
    out.x = lhs.x * rhs.x + lhs.y * rhs.y;
    out.y = lhs.x * rhs.y - lhs.y * rhs.x;
    return out;
}

#define MUL(a, b) __cmul(a, b)

#if IS_CONJ
#define CMUL(a, b) __ccmul(a, b)
#else
#define CMUL(a, b) __cmul(a, b)
#endif

#else
#define MUL(a, b) (a) * (b)
#define CMUL(a, b) (a) * (b)
#endif

// This kernel expects the dense matrix to be transpose of column major (aka non
// transpose row major).

// In this kernel, each block performs multiple "dot" operations.
// In each outer facing iteration, the group performs a "dot" on (one sparse
// row, `THREADS_PER_GROUP` dense columns). The threads in the block load the
// sparse row into local memmory and then perform individual "dot" operations.

kernel void csrmm_nt(global T *output, __global const T *values,
                       global const int *rowidx, __global const int *colidx,
                       const int M, const int N, global const T *rhs,
                       const KParam rinfo, const T alpha, const T beta,
                       global int *counter) {
    int gidx = get_global_id(0);
    int lid  = get_local_id(0);

    rhs += gidx + rinfo.offset;
    output += gidx * M;

    bool within_N = (gidx < N);

    local T s_values[THREADS_PER_GROUP];
    local int s_colidx[THREADS_PER_GROUP];

    int rowNext = get_group_id(1);
    local int s_rowId;

    // Each iteration writes `THREADS_PER_GROUP` columns from one row of the
    // output
    while (true) {
#if USE_GREEDY
        // If the hardware has decent atomic operation support, greediy get the
        // next available row
        if (lid == 0) { s_rowId = atomic_inc(counter + get_group_id(0)); }
        barrier(CLK_LOCAL_MEM_FENCE);
        int rowId = s_rowId;
#else
        // Fall back to the naive distribution of rows otherwise
        int rowId = rowNext;
        rowNext += get_num_groups(1);
#endif
        if (rowId >= M) return;

        // Load the nonzero column offsets for current row
        const int colStart = rowidx[rowId];
        const int colEnd   = rowidx[rowId + 1];

        T outval = 0;
        // Since the number of nonzero elements might be greater than local
        // memory available, Load only part of the row into local memory,
        // perform partial dot, repeat until done.
        for (int id = colStart; id < colEnd; id += THREADS_PER_GROUP) {
            // Load the current chunk of the row into local memory
            int lim       = min(colEnd - id, THREADS_PER_GROUP);
            s_values[lid] = lid < lim ? values[id + lid] : 0;
            s_colidx[lid] = lid < lim ? colidx[id + lid] : -1;
            barrier(CLK_LOCAL_MEM_FENCE);

            // Perform partial "dot" operation for each thread
            for (int idy = 0; within_N && idy < lim; idy++) {
                outval +=
                    CMUL(s_values[idy], rhs[rinfo.strides[1] * s_colidx[idy]]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (within_N) {
            // Each thread writes the output for one column in the current row
#if USE_ALPHA
            outval = MUL(alpha, outval);
#endif

#if USE_BETA
            output[rowId] = outval + MUL(beta, output[rowId]);
#else
            output[rowId] = outval;
#endif
        }
    }
}
