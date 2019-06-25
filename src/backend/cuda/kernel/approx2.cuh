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

namespace cuda {

template<typename Ty, typename Tp, int order>
__global__
void approx2(Param<Ty> zo, CParam<Ty> zi, CParam<Tp> xo,
             const int xdim, const Tp xi_beg,
             const Tp xi_step, CParam<Tp> yo, const int ydim,
             const Tp yi_beg, const Tp yi_step,
             const float offGrid, const int blocksMatX,
             const int blocksMatY, const bool batch,
             InterpolationType method) {
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

    bool is_xo_off[] = {xo.dims[0] > 1, xo.dims[1] > 1, xo.dims[2] > 1,
                        xo.dims[3] > 1};
    bool is_zi_off[] = {true, true, true, true};
    is_zi_off[xdim]  = false;
    is_zi_off[ydim]  = false;

    const int zo_idx =
        idw * zo.strides[3] + idz * zo.strides[2] + idy * zo.strides[1] + idx;
    int xo_idx = idy * xo.strides[1] * is_xo_off[1] + idx * is_xo_off[0];
    int yo_idx = idy * yo.strides[1] * is_xo_off[1] + idx * is_xo_off[0];
    xo_idx +=
        idw * xo.strides[3] * is_xo_off[3] + idz * xo.strides[2] * is_xo_off[2];
    yo_idx +=
        idw * yo.strides[3] * is_xo_off[3] + idz * yo.strides[2] * is_xo_off[2];

    const Tp x = (xo.ptr[xo_idx] - xi_beg) / xi_step;
    const Tp y = (yo.ptr[yo_idx] - yi_beg) / yi_step;
    if (x < 0 || y < 0 || zi.dims[xdim] < x + 1 || zi.dims[ydim] < y + 1) {
        zo.ptr[zo_idx] = scalar<Ty>(offGrid);
        return;
    }

    int zi_idx = idy * zi.strides[1] * is_zi_off[1] + idx * is_zi_off[0];
    zi_idx +=
        idw * zi.strides[3] * is_zi_off[3] + idz * zi.strides[2] * is_zi_off[2];

    // FIXME: Only cubic interpolation is doing clamping
    // We need to make it consistent across all methods
    // Not changing the behavior because tests will fail
    bool clamp = order == 3;

    Interp2<Ty, Tp, order> interp;
    interp(zo, zo_idx, zi, zi_idx, x, y, method, 1, clamp, xdim, ydim);
}

}
