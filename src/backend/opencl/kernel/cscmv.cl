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

T __ccmul(T lhs, T rhs)
{
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

int binary_search(__global int *ptr, int len, int val)
{
    int start = 0;
    int end   = len;
    while (end > start) {
        int mid = start + (end - start) / 2;
        if (val < ptr[mid]) {
            end = mid;
        } else if (val > ptr[mid]) {
            start = mid + 1;
        } else {
            return mid;
        }
    }
    return start;
}

// Each thread performs Matrix Vector multiplications for ROWS_PER_GROUP rows and (K / THREAD) columns.
// This generates a local output buffer of size ROWS_PER_THREAD for each thread.
// The outputs from each thread are added up to generate the final result.
__kernel void
cscmv_block(__global T *output,
            __global const T *values,
            __global const int *colidx,  // rowidx from csr is colidx in csc
            __global const int *rowidx,  // colidx from csr is rowidx in csc
            const int M,                 // K from csr is M in csc
            const int K,                 // M from csr is K in csc
            __global const T *rhs,
            const KParam rinfo,
            const T alpha,
            const T beta)
{
    int lid = get_local_id(0);
    int loff = lid * ROWS_PER_GROUP;
    __local T s_outvals[THREADS * ROWS_PER_GROUP];

    // Get the row offset for the current group in the uncompressed matrix
    int rowOff = get_group_id(0) * ROWS_PER_GROUP;

    for (int i = 0; i < ROWS_PER_GROUP; i++) {
        s_outvals[loff + i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int colId = lid; colId < K; colId += THREADS) {

        int rowStart = colidx[colId];
        int rowEnd   = colidx[colId + 1];
        int nonZeroCount = rowEnd - rowStart;

        // Find the location of the next non zero element after rowOff
        int rowPos   = binary_search(rowidx + rowStart, nonZeroCount, rowOff);
        T rhsval = rhs[colId];

        for (int id =  rowPos + rowStart; id < rowEnd; id++) {
            int rowId = rowidx[id];

            // This work will be done by next few blocks
            if (rowId >= rowOff + ROWS_PER_GROUP) break;

            s_outvals[loff + rowId - rowOff] += CMUL(values[id], rhsval);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Each thread is adding up values from one row at a time.
    for (int n = lid; n < ROWS_PER_GROUP; n += THREADS) {
        if (rowOff + n >= M) break;
        T outval = 0;
        // Add up the partial results from all the threads.
        for (int i = 0; i < THREADS; i++) {
            outval += s_outvals[i * ROWS_PER_GROUP + n];
        }

#if USE_ALPHA
        outval = MUL(alpha, outval);
#endif

#if USE_BETA
        output[rowOff + n] = outval + MUL(beta, output[rowOff + n]);
#else
        output[rowOff + n] = outval;
#endif
    }
}
