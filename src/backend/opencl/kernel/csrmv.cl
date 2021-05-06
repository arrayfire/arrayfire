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

// In this kernel, each thread performs one "dot" operation by reading nonzero
// elements from one row and multiplying with the corresponding elements from
// the dense vector to produce a single output value. This kernel should be used
// when the number of nonzero elements per block is fairly small
kernel void csrmv_thread(global T *output, __global const T *values,
                           global const int *rowidx,
                           global const int *colidx, const int M,
                           global const T *rhs, const KParam rinfo,
                           const T alpha, const T beta
#if USE_GREEDY
                           , global int *counter
#endif
                           ) {
    rhs += rinfo.offset;
    int rowNext = get_global_id(0);

    while (true) {
        // Each thread performs multiple "dot" operations
#if USE_GREEDY
        // Considering that the number of non zero elements per row can be
        // uneven a greedy approach may be useful. This acheived by getting the
        // next available row to perform the "dot" operation on.
        int rowId = atomic_inc(counter);
#else
        // Unfortunately atomic operations are costly on some architectures.
        // The fallback is to use same number of rows on all threads.
        int rowId = rowNext;
        rowNext += get_global_size(0);
#endif
        if (rowId >= M) return;

        // Find the columns offsets for the current row
        int colStart = rowidx[rowId];
        int colEnd   = rowidx[rowId + 1];

        T outval = 0;
        // Performing the "dot" operation
        for (int id = colStart; id < colEnd; id++) {
            int cid = colidx[id];
            outval += CMUL(values[id], rhs[cid]);
        }

        // Writing out a single output
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

// In this kernel, each block performs one "dot" operation by having each thread
// read a nonzero element from a row and multiplying with the corresponding
// elements from dense vector to produce a local output values. Then the block
// performs a reduction operation to produce a single output value. This kernel
// should be used when the number of nonzero elements per block is large
kernel void csrmv_block(global T *output, __global const T *values,
                          global const int *rowidx,
                          global const int *colidx, const int M,
                          global const T *rhs, const KParam rinfo,
                          const T alpha, const T beta
#if USE_GREEDY
                          , global int *counter
#endif
                          ) {
    rhs += rinfo.offset;
    int lid     = get_local_id(0);
    int rowNext = get_group_id(0);
    local int s_rowId;

    // Each thread stores part of the output result
    local T s_outval[THREADS];

    // Each groups performs multiple "dot" operations
    while (true) {
#if USE_GREEDY
        // Considering that the number of non zero elements per row can be
        // uneven a greedy approach may be useful. This acheived by getting the
        // next available row to perform the "dot" operation on. Since the rowId
        // needs is the same across the block, only one thread needs to
        // increment the counter.
        if (lid == 0) { s_rowId = atomic_inc(counter); }
        barrier(CLK_LOCAL_MEM_FENCE);
        int rowId = s_rowId;
#else
        // Unfortunately atomic operations are costly on some architectures.
        // The fallback is to use same number of rows on all blocks.
        int rowId = rowNext;
        rowNext += get_num_groups(0);
#endif
        if (rowId >= M) return;

        int colStart = rowidx[rowId];
        int colEnd   = rowidx[rowId + 1];
        T outval     = 0;

        // Each thread performs "dot" on num_nonzero_elements / THREADS for a
        // given row
        for (int id = colStart + lid; id < colEnd; id += THREADS) {
            int cid = colidx[id];
            outval += MUL(values[id], rhs[cid]);
        }
        s_outval[lid] = outval;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform a block reduce operation to get the single output value
        for (int n = THREADS / 2; n > 0; n /= 2) {
            if (lid < n) s_outval[lid] += s_outval[lid + n];
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // A single thread writes the output value
        if (lid == 0) {
#if USE_ALPHA
            outval = MUL(alpha, s_outval[0]);
#else
            outval        = s_outval[0];
#endif

#if USE_BETA
            output[rowId] = outval + MUL(beta, output[rowId]);
#else
            output[rowId] = outval;
#endif
        }
    }
}
