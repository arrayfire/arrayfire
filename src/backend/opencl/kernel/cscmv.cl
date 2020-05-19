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

int binary_search(global const int *ptr, int len, int val) {
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

// Each thread performs Matrix Vector multiplications for ROWS_PER_GROUP rows
// and (K / THREAD) columns. This generates a local output buffer of size
// ROWS_PER_THREAD for each thread. The outputs from each thread are added up to
// generate the final result.
kernel void cscmv_block(
    global T *output, __global const T *values,
    global const int *colidx,  // rowidx from csr is colidx in csc
    global const int *rowidx,  // colidx from csr is rowidx in csc
    const int M,                 // K from csr is M in csc
    const int K,                 // M from csr is K in csc
    global const T *rhs, const KParam rinfo, const T alpha, const T beta) {
    int lid = get_local_id(0);

    // Get the row offset for the current group in the uncompressed matrix
    int rowOff = get_group_id(0) * ROWS_PER_GROUP;
    int rowLim = min(ROWS_PER_GROUP, M - rowOff);
    rhs += rinfo.offset;

    T l_outvals[ROWS_PER_GROUP];
    for (int i = 0; i < rowLim; i++) { l_outvals[i] = 0; }

    for (int colId = lid; colId < K; colId += THREADS) {
        int rowStart     = colidx[colId];
        int rowEnd       = colidx[colId + 1];
        int nonZeroCount = rowEnd - rowStart;

        // Find the location of the next non zero element after rowOff
        int rowPos = binary_search(rowidx + rowStart, nonZeroCount, rowOff);
        T rhsval   = rhs[colId];

        // Traversing through nonzero elements in the current chunk
        for (int id = rowPos + rowStart; id < rowEnd; id++) {
            int rowId = rowidx[id];

            // Exit if moving past current chunk
            if (rowId >= rowOff + ROWS_PER_GROUP) break;

            l_outvals[rowId - rowOff] += CMUL(values[id], rhsval);
        }
    }

    // s_outvals is used for reduction
    local T s_outvals[THREADS];

    // s_output is used to store the final output into local memory
    local T s_output[ROWS_PER_GROUP];

    // For each row of output, copy registers to local memory, add results,
    // write to output.
    for (int i = 0; i < rowLim; i++) {
        // Copying to local memory
        s_outvals[lid] = l_outvals[i];
        barrier(CLK_LOCAL_MEM_FENCE);

        // Adding the results through reduction
        for (int n = THREADS / 2; n > 0; n /= 2) {
            if (lid < n) s_outvals[lid] += s_outvals[lid + n];
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Store to another local buffer so it can be written in a coalesced
        // manner later
        if (lid == 0) { s_output[i] = s_outvals[0]; }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // For each row in output, write output in coalesced manner
    for (int i = lid; i < ROWS_PER_GROUP; i += THREADS) {
        T outval = s_output[i];

#if USE_ALPHA
        outval = MUL(alpha, outval);
#endif

#if USE_BETA
        output[rowOff + i] = outval + MUL(beta, output[j * M + rowOff + i]);
#else
        output[rowOff + i] = outval;
#endif
    }
}
