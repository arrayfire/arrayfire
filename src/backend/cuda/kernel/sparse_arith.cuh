/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <math.hpp>
#include <optypes.hpp>

namespace arrayfire {
namespace cuda {

template<typename T, af_op_t op>
struct arith_op {
    T operator()(T v1, T v2) { return T(0); }
};

template<typename T>
struct arith_op<T, af_add_t> {
    T operator()(T v1, T v2) { return v1 + v2; }
};

template<typename T>
struct arith_op<T, af_sub_t> {
    T operator()(T v1, T v2) { return v1 - v2; }
};

template<typename T>
struct arith_op<T, af_mul_t> {
    T operator()(T v1, T v2) { return v1 * v2; }
};

template<typename T>
struct arith_op<T, af_div_t> {
    T operator()(T v1, T v2) { return v1 / v2; }
};

// All Kernels follow below naming convention
// <format>ArithXYZ where
// <format> is either csr or coo
// X - D for Dense output, S for sparse output
// Y - D for Dense lhs, S for sparse lhs
// Z - D for Dense rhs, S for sparse rhs

template<typename T, af_op_t op>
__global__ void csrArithDSD(Param<T> out, CParam<T> values, CParam<int> rowIdx,
                            CParam<int> colIdx, CParam<T> rhs,
                            const bool reverse) {
    const int row = blockIdx.x * TY + threadIdx.y;

    if (row >= out.dims[0]) return;

    const int rowStartIdx = rowIdx.ptr[row];
    const int rowEndIdx   = rowIdx.ptr[row + 1];

    // Repeat loop until all values in the row are computed
    for (int idx = rowStartIdx + threadIdx.x; idx < rowEndIdx; idx += TX) {
        const int col = colIdx.ptr[idx];

        if (row >= out.dims[0] || col >= out.dims[1]) continue;  // Bad indices

        // Get Values
        const T val  = values.ptr[idx];
        const T rval = rhs.ptr[col * rhs.strides[1] + row];

        const int offset = col * out.strides[1] + row;
        if (reverse)
            out.ptr[offset] = arith_op<T, op>()(rval, val);
        else
            out.ptr[offset] = arith_op<T, op>()(val, rval);
    }
}

template<typename T, af_op_t op>
__global__ void cooArithDSD(Param<T> out, CParam<T> values, CParam<int> rowIdx,
                            CParam<int> colIdx, CParam<T> rhs,
                            const bool reverse) {
    const int idx = blockIdx.x * THREADS + threadIdx.x;

    if (idx >= values.dims[0]) return;

    const int row = rowIdx.ptr[idx];
    const int col = colIdx.ptr[idx];

    if (row >= out.dims[0] || col >= out.dims[1]) return;  // Bad indices

    // Get Values
    const T val  = values.ptr[idx];
    const T rval = rhs.ptr[col * rhs.strides[1] + row];

    const int offset = col * out.strides[1] + row;
    if (reverse)
        out.ptr[offset] = arith_op<T, op>()(rval, val);
    else
        out.ptr[offset] = arith_op<T, op>()(val, rval);
}

template<typename T, af_op_t op>
__global__ void csrArithSSD(Param<T> values, Param<int> rowIdx,
                            Param<int> colIdx, CParam<T> rhs,
                            const bool reverse) {
    const int row = blockIdx.x * TY + threadIdx.y;

    if (row >= rhs.dims[0]) return;

    const int rowStartIdx = rowIdx.ptr[row];
    const int rowEndIdx   = rowIdx.ptr[row + 1];

    // Repeat loop until all values in the row are computed
    for (int idx = rowStartIdx + threadIdx.x; idx < rowEndIdx; idx += TX) {
        const int col = colIdx.ptr[idx];

        if (row >= rhs.dims[0] || col >= rhs.dims[1]) continue;  // Bad indices

        // Get Values
        const T val  = values.ptr[idx];
        const T rval = rhs.ptr[col * rhs.strides[1] + row];

        if (reverse)
            values.ptr[idx] = arith_op<T, op>()(rval, val);
        else
            values.ptr[idx] = arith_op<T, op>()(val, rval);
    }
}

template<typename T, af_op_t op>
__global__ void cooArithSSD(Param<T> values, Param<int> rowIdx,
                            Param<int> colIdx, CParam<T> rhs,
                            const bool reverse) {
    const int idx = blockIdx.x * THREADS + threadIdx.x;

    if (idx >= values.dims[0]) return;

    const int row = rowIdx.ptr[idx];
    const int col = colIdx.ptr[idx];

    if (row >= rhs.dims[0] || col >= rhs.dims[1]) return;  // Bad indices

    // Get Values
    const T val  = values.ptr[idx];
    const T rval = rhs.ptr[col * rhs.strides[1] + row];

    if (reverse)
        values.ptr[idx] = arith_op<T, op>()(rval, val);
    else
        values.ptr[idx] = arith_op<T, op>()(val, rval);
}

}  // namespace cuda
}  // namespace arrayfire
