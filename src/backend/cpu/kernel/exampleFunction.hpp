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
void exampleFunction(Array<T> out, Array<T> const a, Array<T> const b, const af_someenum_t method)
{
    dim4 oDims    = out.dims();

    dim4 aStrides = a.strides();        // you can retrieve strides
    dim4 bStrides = b.strides();
    dim4 oStrides = out.strides();

    const T* src1 = a.get();            // cpu::Array<T>::get returns the pointer to the
                                        // memory allocated for that Array (with proper offsets)
    const T* src2 = b.get();            // cpu::Array<T>::get returns the pointer to the
                                        // memory allocated for that Array (with proper offsets)
    T* dst = out.get();

    // Implement your algorithm and write results to dst
    for(int j=0; j<oDims[1]; ++j) {
        for (int i=0; i<oDims[0]; ++i) {

            int src1Idx = i + j*aStrides[1];
            int src2Idx = i + j*bStrides[1];
            int dstIdx  = i + j*oStrides[1];

            // kernel algorithm goes here
            dst[dstIdx] = src1[src1Idx] + src2[src2Idx];
        }
    }
}

}
}
