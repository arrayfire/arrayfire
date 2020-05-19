/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

kernel void sparseArithCOO(global T *oPtr, const KParam out,
                           global const T *values, global const int *rowIdx,
                           global const int *colIdx, const int nNZ,
                           global const T *rPtr, const KParam rhs,
                           const int reverse) {
    const int idx = get_global_id(0);

    if (idx >= nNZ) return;

    const int row = rowIdx[idx];
    const int col = colIdx[idx];

    if (row >= out.dims[0] || col >= out.dims[1]) return;  // Bad indices

    // Get Values
    const T val  = values[idx];
    const T rval = rPtr[col * rhs.strides[1] + row];

    const int offset = col * out.strides[1] + row;
    if (reverse)
        oPtr[offset] = OP(rval, val);
    else
        oPtr[offset] = OP(val, rval);
}

kernel void sparseArithCOO2(global T *values, global int *rowIdx,
                            global int *colIdx, const int nNZ,
                            global const T *rPtr, const KParam rhs,
                            const int reverse) {
    const int idx = get_global_id(0);

    if (idx >= nNZ) return;

    const int row = rowIdx[idx];
    const int col = colIdx[idx];

    if (row >= rhs.dims[0] || col >= rhs.dims[1]) return;  // Bad indices

    // Get Values
    const T val  = values[idx];
    const T rval = rPtr[col * rhs.strides[1] + row];

    if (reverse)
        values[idx] = OP(rval, val);
    else
        values[idx] = OP(val, rval);
}
