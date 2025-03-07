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

template<typename T, bool is_upper, bool is_unit_diag>
__global__ void triangle(Param<T> r, CParam<T> in, const int blocksPerMatX,
                         const int blocksPerMatY) {
    const int oz = blockIdx.x / blocksPerMatX;
    const int ow = (blockIdx.y + blockIdx.z * gridDim.y) / blocksPerMatY;

    const int blockIdx_x = blockIdx.x - oz * blocksPerMatX;
    const int blockIdx_y =
        (blockIdx.y + blockIdx.z * gridDim.y) - ow * blocksPerMatY;

    const int xx = threadIdx.x + blockIdx_x * blockDim.x;
    const int yy = threadIdx.y + blockIdx_y * blockDim.y;

    const int incy = blocksPerMatY * blockDim.y;
    const int incx = blocksPerMatX * blockDim.x;

    T *d_r       = r.ptr;
    const T *d_i = in.ptr;

    const T one  = scalar<T>(1);
    const T zero = scalar<T>(0);

    if (oz < r.dims[2] && ow < r.dims[3]) {
        d_i = d_i + oz * in.strides[2] + ow * in.strides[3];
        d_r = d_r + oz * r.strides[2] + ow * r.strides[3];

        for (int oy = yy; oy < r.dims[1]; oy += incy) {
            const T *Yd_i = d_i + oy * in.strides[1];
            T *Yd_r       = d_r + oy * r.strides[1];

            for (int ox = xx; ox < r.dims[0]; ox += incx) {
                bool cond         = is_upper ? (oy >= ox) : (oy <= ox);
                bool do_unit_diag = is_unit_diag && (ox == oy);
                if (cond) {
                    // Change made because of compute 53 failing tests
                    Yd_r[ox] = do_unit_diag ? one : Yd_i[ox];
                } else {
                    Yd_r[ox] = zero;
                }
            }
        }
    }
}

}  // namespace cuda
}  // namespace arrayfire
