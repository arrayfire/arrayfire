/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

__kernel
void sparse_arith_coo_kernel(__global T *oPtr,
                             const KParam out,
                             __global const T *values,
                             __global const int *rowIdx,
                             __global const int *colIdx,
                             const int nNZ,
                             __global const T *rPtr,
                             const KParam rhs,
                             const int reverse)
{
    const int idx = get_global_id(0);

    if(idx >= nNZ) return;

    const int row = rowIdx[idx];
    const int col = colIdx[idx];

    if(row >= out.dims[0] || col >= out.dims[1]) return;    // Bad indices

    // Get Values
    const T val  = values[idx];
    const T rval = rPtr[col * rhs.strides[1] + row];

    const int offset = col * out.strides[1] + row;
    if(reverse) oPtr[offset] = OP(rval, val);
    else        oPtr[offset] = OP(val, rval);
}

__kernel
void sparse_arith_coo_kernel_S(__global T *values,
                               __global int *rowIdx,
                               __global int *colIdx,
                               const int nNZ,
                               __global const T *rPtr,
                               const KParam rhs,
                               const int reverse)
{
    const int idx = get_global_id(0);

    if(idx >= nNZ) return;

    const int row = rowIdx[idx];
    const int col = colIdx[idx];

    if(row >= rhs.dims[0] || col >= rhs.dims[1]) return;    // Bad indices

    // Get Values
    const T val  = values[idx];
    const T rval = rPtr[col * rhs.strides[1] + row];

    if(reverse) values[idx] = OP(rval, val);
    else        values[idx] = OP(val, rval);
}
