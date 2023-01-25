/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Param.hpp>
#include <math.hpp>

namespace arrayfire {
namespace cuda {

template<typename T>
__global__ void exampleFunc(Param<T> c, CParam<T> a, CParam<T> b,
                            const af_someenum_t p) {
    // get current thread global identifiers along required dimensions
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < a.dims[0] && j < a.dims[1]) {
        // if needed use strides array to compute linear index of arrays
        int src1Idx = i + j * a.strides[1];
        int src2Idx = i + j * b.strides[1];
        int dstIdx  = i + j * c.strides[1];

        T* dst        = c.ptr;
        const T* src1 = a.ptr;
        const T* src2 = b.ptr;

        // kernel algorithm goes here
        dst[dstIdx] = src1[src1Idx] + src2[src2Idx];
    }
}

}  // namespace cuda
}  // namespace arrayfire
