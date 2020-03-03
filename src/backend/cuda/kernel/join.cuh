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

template<typename To, typename Ti, int dim>
__global__ void join(Param<To> out, CParam<Ti> in, const int o0, const int o1,
                     const int o2, const int o3, const int blocksPerMatX,
                     const int blocksPerMatY) {
    const int incy = blocksPerMatY * blockDim.y;
    const int incx = blocksPerMatX * blockDim.x;

    const int iz         = blockIdx.x / blocksPerMatX;
    const int blockIdx_x = blockIdx.x - iz * blocksPerMatX;
    const int xx         = threadIdx.x + blockIdx_x * blockDim.x;

    To *d_out      = out.ptr;
    Ti const *d_in = in.ptr;

    const int iw = (blockIdx.y + (blockIdx.z * gridDim.y)) / blocksPerMatY;
    const int blockIdx_y =
        (blockIdx.y + (blockIdx.z * gridDim.y)) - iw * blocksPerMatY;
    const int yy = threadIdx.y + blockIdx_y * blockDim.y;

    if (iz < in.dims[2] && iw < in.dims[3]) {
        d_out = d_out + (iz + o2) * out.strides[2] + (iw + o3) * out.strides[3];
        d_in  = d_in + iz * in.strides[2] + iw * in.strides[3];

        for (int iy = yy; iy < in.dims[1]; iy += incy) {
            Ti const *d_in_ = d_in + iy * in.strides[1];
            To *d_out_      = d_out + (iy + o1) * out.strides[1];

            for (int ix = xx; ix < in.dims[0]; ix += incx) {
                d_out_[ix + o0] = d_in_[ix];
            }
        }
    }
}

}  // namespace cuda
