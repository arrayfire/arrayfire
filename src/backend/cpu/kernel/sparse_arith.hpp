/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Array.hpp>
#include <math.hpp>

namespace cpu
{
namespace kernel
{

template<typename T, af_op_t op>
struct arith_op
{
    T operator()(T v1, T v2)
    {
        return scalar<T>(0);
    }
};

template<typename T>
struct arith_op<T, af_add_t>
{
    T operator()(T v1, T v2)
    {
        return v1 + v2;
    }
};

template<typename T>
struct arith_op<T, af_sub_t>
{
    T operator()(T v1, T v2)
    {
        return v1 - v2;
    }
};

template<typename T>
struct arith_op<T, af_mul_t>
{
    T operator()(T v1, T v2)
    {
        return v1 * v2;
    }
};

template<typename T>
struct arith_op<T, af_div_t>
{
    T operator()(T v1, T v2)
    {
        return v1 / v2;
    }
};

template<typename T, af_op_t op, af_storage type>
void sparseArithOp(Array<T> output,
                   const Array<T> values, const Array<int> rowIdx, const Array<int> colIdx,
                   const Array<T> rhs, const bool reverse = false)
{
    T * oPtr = output.get();
    const T   * hPtr = rhs.get();

    const T   * vPtr = values.get();
    const int * rPtr = rowIdx.get();
    const int * cPtr = colIdx.get();

    dim4 odims    = output.dims();
    dim4 ostrides = output.strides();
    dim4 hstrides = rhs.strides();

    std::vector<int> temp;
    if(type == AF_STORAGE_CSR) {
        temp.resize(values.elements());
        for(int i = 0; i < rowIdx.dims()[0] - 1; i++) {
            for(int ii = rPtr[i]; ii < rPtr[i + 1]; ii++) {
                temp[ii] = i;
            }
        }
    //} else if(type == AF_STORAGE_CSC) {   // For future
    }

    const int *xx = (type == AF_STORAGE_CSR) ? temp.data() : rPtr;
    const int *yy = (type == AF_STORAGE_CSC) ? temp.data() : cPtr;

    for(int i = 0; i < (int)values.elements(); i++) {
        // Bad index data
        if(xx[i] >= odims[0] || yy [i]>= odims[1]) continue;

        int offset = xx[i] + yy[i] * ostrides[1];
        int hoff   = xx[i] + yy[i] * hstrides[1];

        if(reverse) oPtr[offset] = arith_op<T, op>()(hPtr[hoff], vPtr[i]);
        else        oPtr[offset] = arith_op<T, op>()(vPtr[i], hPtr[hoff]);
    }
}

}
}

