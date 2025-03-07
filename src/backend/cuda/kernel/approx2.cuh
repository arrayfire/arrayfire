/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <interp.hpp>

namespace arrayfire {
namespace cuda {

template<typename Ty, typename Tp, int xdim, int ydim, int order>
__global__ void approx2(Param<Ty> zo, CParam<Ty> zi, CParam<Tp> xo,
                        const Tp xi_beg, const Tp xi_step_reproc, CParam<Tp> yo,
                        const Tp yi_beg, const Tp yi_step_reproc,
                        const float offGrid, const int blocksMatX,
                        const int blocksMatY, const bool batch,
                        af::interpType method) {
    const int idz        = blockIdx.x / blocksMatX;
    const int blockIdx_x = blockIdx.x - idz * blocksMatX;
    const int idx        = threadIdx.x + blockIdx_x * blockDim.x;

    const int idw = (blockIdx.y + blockIdx.z * gridDim.y) / blocksMatY;
    const int blockIdx_y =
        (blockIdx.y + blockIdx.z * gridDim.y) - idw * blocksMatY;
    const int idy = threadIdx.y + blockIdx_y * blockDim.y;

    if (idx >= zo.dims[0] || idy >= zo.dims[1] || idz >= zo.dims[2] ||
        idw >= zo.dims[3])
        return;

    // FIXME: Only cubic interpolation is doing clamping
    // We need to make it consistent across all methods
    // Not changing the behavior because tests will fail
    const bool clamp = order == 3;

    bool is_off[] = {xo.dims[0] > 1, xo.dims[1] > 1, xo.dims[2] > 1,
                     xo.dims[3] > 1};

    const int zo_idx =
        idw * zo.strides[3] + idz * zo.strides[2] + idy * zo.strides[1] + idx;
    int xo_idx = idy * xo.strides[1] * is_off[1] + idx * is_off[0];
    int yo_idx = idy * yo.strides[1] * is_off[1] + idx * is_off[0];
    if (batch) {
        xo_idx +=
            idw * xo.strides[3] * is_off[3] + idz * xo.strides[2] * is_off[2];
        yo_idx +=
            idw * yo.strides[3] * is_off[3] + idz * yo.strides[2] * is_off[2];
    }

    const Tp x = (xo.ptr[xo_idx] - xi_beg) * xi_step_reproc;
    const Tp y = (yo.ptr[yo_idx] - yi_beg) * yi_step_reproc;

#pragma unroll
    for (int flagIdx = 0; flagIdx < 4; ++flagIdx) { is_off[flagIdx] = true; }
    is_off[xdim] = false;
    is_off[ydim] = false;

    if (x < 0 || y < 0 || zi.dims[xdim] < x + 1 || zi.dims[ydim] < y + 1) {
        zo.ptr[zo_idx] = scalar<Ty>(offGrid);
        return;
    }

    int zi_idx = idy * zi.strides[1] * is_off[1] + idx * is_off[0];
    zi_idx += idw * zi.strides[3] * is_off[3] + idz * zi.strides[2] * is_off[2];

    Interp2<Ty, Tp, xdim, ydim, order> interp;
    interp(zo, zo_idx, zi, zi_idx, x, y, method, 1, clamp);
}

}  // namespace cuda
}  // namespace arrayfire
