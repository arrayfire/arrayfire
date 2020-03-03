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

namespace cuda {

template<typename T>
__global__ void memcopy(Param<T> out, CParam<T> in, uint blocks_x,
                        uint blocks_y) {
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int zid        = blockIdx.x / blocks_x;
    const int blockIdx_x = blockIdx.x - (blocks_x)*zid;
    const int xid        = blockIdx_x * blockDim.x + tidx;

    const int wid = (blockIdx.y + blockIdx.z * gridDim.y) / blocks_y;
    const int blockIdx_y =
        (blockIdx.y + blockIdx.z * gridDim.y) - (blocks_y)*wid;
    const int yid = blockIdx_y * blockDim.y + tidy;
    // FIXME: Do more work per block
    T *const optr = out.ptr + wid * out.strides[3] + zid * out.strides[2] +
                    yid * out.strides[1];
    const T *iptr = in.ptr + wid * in.strides[3] + zid * in.strides[2] +
                    yid * in.strides[1];

    int istride0 = in.strides[0];
    if (xid < in.dims[0] && yid < in.dims[1] && zid < in.dims[2] &&
        wid < in.dims[3]) {
        optr[xid] = iptr[xid * istride0];
    }
}

}  // namespace cuda
