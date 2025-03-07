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
__global__ void iota(Param<T> out, const int s0, const int s1, const int s2,
                     const int s3, const int blocksPerMatX,
                     const int blocksPerMatY) {
    const int oz         = blockIdx.x / blocksPerMatX;
    const int blockIdx_x = blockIdx.x - oz * blocksPerMatX;
    const int xx         = threadIdx.x + blockIdx_x * blockDim.x;

    const int ow = (blockIdx.y + blockIdx.z * gridDim.y) / blocksPerMatY;
    const int blockIdx_y =
        (blockIdx.y + blockIdx.z * gridDim.y) - ow * blocksPerMatY;
    const int yy = threadIdx.y + blockIdx_y * blockDim.y;

    if (xx >= out.dims[0] || yy >= out.dims[1] || oz >= out.dims[2] ||
        ow >= out.dims[3])
        return;

    const int ozw = ow * out.strides[3] + oz * out.strides[2];

    dim_t val = (ow % s3) * s2 * s1 * s0;
    val += (oz % s2) * s1 * s0;

    const int incy = blocksPerMatY * blockDim.y;
    const int incx = blocksPerMatX * blockDim.x;

    for (int oy = yy; oy < out.dims[1]; oy += incy) {
        int oyzw   = ozw + oy * out.strides[1];
        dim_t valY = val + (oy % s1) * s0;
        for (int ox = xx; ox < out.dims[0]; ox += incx) {
            int oidx = oyzw + ox;

            out.ptr[oidx] = valY + (ox % s0);
        }
    }
}

}  // namespace cuda
}  // namespace arrayfire
