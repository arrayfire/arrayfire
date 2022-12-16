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
__global__ void createDiagonalMat(Param<T> out, CParam<T> in, int num,
                                  int blocks_x) {
    unsigned idz        = blockIdx.x / blocks_x;
    unsigned blockIdx_x = blockIdx.x - idz * blocks_x;

    unsigned idx = threadIdx.x + blockIdx_x * blockDim.x;
    unsigned idy =
        threadIdx.y + (blockIdx.y + blockIdx.z * gridDim.y) * blockDim.y;

    if (idx >= out.dims[0] || idy >= out.dims[1] || idz >= out.dims[2]) return;

    T *optr       = out.ptr + idz * out.strides[2] + idy * out.strides[1] + idx;
    const T *iptr = in.ptr + idz * in.strides[1] + ((num > 0) ? idx : idy);

    T val = (idx == (idy - num)) ? *iptr : scalar<T>(0);
    *optr = val;
}

template<typename T>
__global__ void extractDiagonal(Param<T> out, CParam<T> in, int num,
                                int blocks_z) {
    unsigned idw = (blockIdx.y + blockIdx.z * gridDim.y) / blocks_z;
    unsigned idz = (blockIdx.y + blockIdx.z * gridDim.y) - idw * blocks_z;

    unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= out.dims[0] || idz >= out.dims[2] || idw >= out.dims[3]) return;

    T *optr = out.ptr + idz * out.strides[2] + idw * out.strides[3] + idx;

    if (idx >= in.dims[0] || idx >= in.dims[1]) *optr = scalar<T>(0);

    int i_off     = (num > 0) ? (num * in.strides[1] + idx) : (idx - num);
    const T *iptr = in.ptr + idz * in.strides[2] + idw * in.strides[3] + i_off;
    *optr         = iptr[idx * in.strides[1]];
}

}  // namespace cuda
}  // namespace arrayfire
