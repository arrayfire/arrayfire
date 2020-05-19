/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// TODO_PERF(pradeep) More performance improvements are possible
__attribute__((reqd_work_group_size(256, 1, 1))) kernel void ssarith_csr(
    global T *oVals, global int *oColIdx, global const int *oRowIdx, uint M,
    uint N, uint nnza, global const T *lVals, global const int *lRowIdx,
    global const int *lColIdx, uint nnzb, global const T *rVals,
    global const int *rRowIdx, global const int *rColIdx) {
    const uint row = get_global_id(0);

    const bool valid = row < M;

    const uint lEnd   = (valid ? lRowIdx[row + 1] : 0);
    const uint rEnd   = (valid ? rRowIdx[row + 1] : 0);
    const uint offset = (valid ? oRowIdx[row] : 0);

    global T *ovPtr   = oVals + offset;
    global int *ocPtr = oColIdx + offset;

    uint l = (valid ? lRowIdx[row] : 0);
    uint r = (valid ? rRowIdx[row] : 0);

    uint nnz = 0;
    while (l < lEnd && r < rEnd) {
        uint lci = lColIdx[l];
        uint rci = rColIdx[r];

        T lhs = (lci <= rci ? lVals[l] : (T)(IDENTITY_VALUE));
        T rhs = (lci >= rci ? rVals[r] : (T)(IDENTITY_VALUE));

        ovPtr[nnz] = OP(lhs, rhs);
        ocPtr[nnz] = (lci <= rci) ? lci : rci;

        l += (lci <= rci);
        r += (lci >= rci);
        nnz++;
    }
    while (l < lEnd) {
        ovPtr[nnz] = OP(lVals[l], (T)(IDENTITY_VALUE));
        ocPtr[nnz] = lColIdx[l];
        l++;
        nnz++;
    }
    while (r < rEnd) {
        ovPtr[nnz] = OP((T)(IDENTITY_VALUE), rVals[r]);
        ocPtr[nnz] = rColIdx[r];
        r++;
        nnz++;
    }
}
