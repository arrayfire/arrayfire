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

namespace arrayfire {
namespace cuda {

template<typename T>
__global__ void identity(Param<T> out, int blocks_x, int blocks_y) {
    const dim_t idz = blockIdx.x / blocks_x;
    const dim_t idw = (blockIdx.y + blockIdx.z * gridDim.y) / blocks_y;

    const dim_t blockIdx_x = blockIdx.x - idz * blocks_x;
    const dim_t blockIdx_y =
        (blockIdx.y + blockIdx.z * gridDim.y) - idw * blocks_y;

    const dim_t idx = threadIdx.x + blockIdx_x * blockDim.x;
    const dim_t idy = threadIdx.y + blockIdx_y * blockDim.y;

    if (idx >= out.dims[0] || idy >= out.dims[1] || idz >= out.dims[2] ||
        idw >= out.dims[3])
        return;

    const T one  = scalar<T>(1);
    const T zero = scalar<T>(0);

    T *ptr = out.ptr + idz * out.strides[2] + idw * out.strides[3];
    T val  = (idx == idy) ? one : zero;
    ptr[idx + idy * out.strides[1]] = val;
}

}  // namespace cuda
}  // namespace arrayfire
