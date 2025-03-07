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

namespace arrayfire {
namespace cuda {

template<typename T>
__global__ void coo2Dense(Param<T> output, CParam<T> values, CParam<int> rowIdx,
                          CParam<int> colIdx) {
    for (int i = threadIdx.x; i < reps * blockDim.x; i += blockDim.x) {
        int id = i + blockIdx.x * blockDim.x * reps;
        if (id >= values.dims[0]) return;

        T v   = values.ptr[id];
        int r = rowIdx.ptr[id];
        int c = colIdx.ptr[id];

        int offset = r + c * output.strides[1];

        output.ptr[offset] = v;
    }
}

}  // namespace cuda
}  // namespace arrayfire
