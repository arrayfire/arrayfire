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
#include <utility.hpp>

namespace cpu
{
namespace kernel
{

template<typename T>
void coo2dense(Array<T> output,
               Array<T> const values, Array<int> const rowIdx, Array<int> const colIdx)
{
    T   const * const vPtr = values.get();
    int const * const rPtr = rowIdx.get();
    int const * const cPtr = colIdx.get();

    T * outPtr = output.get();

    af::dim4 ostrides = output.strides();

    int nNZ = values.dims()[0];
    for(int i = 0; i < nNZ; i++) {
        T   v = vPtr[i];
        int r = rPtr[i];
        int c = cPtr[i];

        int offset = r + c * ostrides[1];

        outPtr[offset] = v;
    }
}

}
}
