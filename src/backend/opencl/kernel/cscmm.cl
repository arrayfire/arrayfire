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

// Each group computes an output of size ROWS_PER_GROUP x COLS_PER_GROUP
// Each thread in a group maintains the partial outputs of size ROWS_PER_GROUP x
// COLS_PER_GROUP The outputs from each thread are added up to generate the
// final result.
kernel void cscmm_nn(
    global T *output, __global const T *values,
    global const int *colidx,  // rowidx from csr is colidx in csc
    global const int *rowidx,  // colidx from csr is rowidx in csc
    const int M,                 // K from csr is M in csc
    const int K,                 // M from csr is K in csc
    const int N,                 // N is number of columns in dense matrix
    global const T *rhs, const KParam rinfo, const T alpha, const T beta) {
    int lid = get_local_id(0);

    // Get the row offset for the current group in the uncompressed matrix
    int rowOff = get_group_id(0) * ROWS_PER_GROUP;
    int colOff = get_group_id(1) * COLS_PER_GROUP;

    // Ensure you are not going out of bounds
    int rowLim = min(ROWS_PER_GROUP, M - rowOff);
    int colLim = min(COLS_PER_GROUP, N - colOff);

    rhs += colOff * rinfo.strides[1] + rinfo.offset;
    output += colOff * M;

    // Initialize partial output to 0
    T l_outvals[COLS_PER_GROUP][ROWS_PER_GROUP];
    for (int j = 0; j < colLim; j++) {
        for (int i = 0; i < rowLim; i++) { l_outvals[j][i] = 0; }
    }

    // Dot requires you to traverse the entire inner dimension
    for (int colId = lid; colId < K; colId += THREADS) {
        int rowStart     = colidx[colId];
        int rowEnd       = colidx[colId + 1];
        int nonZeroCount = rowEnd - rowStart;

        // Find the location of the next non zero element after rowOff
        int rowPos = binary_search(rowidx + rowStart, nonZeroCount, rowOff);

        // Read the rhs values from all the columns as they can be reused for
        // all rows
        T rhsvals[COLS_PER_GROUP];
        for (int j = 0; j < colLim; j++) {
            rhsvals[j] = rhs[colId + j * rinfo.strides[1]];
        }

        // Traversing through nonzero elements in the current chunk
        for (int id = rowPos + rowStart; id < rowEnd; id++) {
            int rowId = rowidx[id];

            // Exit if going past current chunk
            if (rowId >= rowOff + ROWS_PER_GROUP) break;

            // Read the row value and multiply with all columns
            T lhsval = values[id];
            for (int j = 0; j < colLim; j++) {
                l_outvals[j][rowId - rowOff] += CMUL(lhsval, rhsvals[j]);
            }
        }
    }

    local T s_outvals[THREADS];

    // For each row and col of output, copy registers to local memory, add
    // results, write to output.
    for (int j = 0; j < colLim; j++) {
        for (int i = 0; i < rowLim; i++) {
            // Copying to local memory
            s_outvals[lid] = l_outvals[j][i];
            barrier(CLK_LOCAL_MEM_FENCE);

            // Adding the results through reduction
            for (int n = THREADS / 2; n > 0; n /= 2) {
                if (lid < n) s_outvals[lid] += s_outvals[lid + n];
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            // Writing to output
            if (lid == 0) {
                T outval = s_outvals[0];

#if USE_ALPHA
                outval = MUL(alpha, outval);
#endif

#if USE_BETA
                output[j * M + rowOff + i] =
                    outval + MUL(beta, output[j * M + rowOff + i]);
#else
                output[j * M + rowOff + i] = outval;
#endif
            }
        }
    }
}
