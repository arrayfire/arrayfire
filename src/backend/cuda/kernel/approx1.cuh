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
void approx1(Param<Ty> yo, CParam<Ty> yi, CParam<Tp> xo,
             const int xdim, const Tp xi_beg,
             const Tp xi_step, const float offGrid,
             const int blocksMatX, const bool batch,
             af::interpType method) {
    const int idy        = blockIdx.x / blocksMatX;
    const int blockIdx_x = blockIdx.x - idy * blocksMatX;
    const int idx        = blockIdx_x * blockDim.x + threadIdx.x;

    const int idw = (blockIdx.y + blockIdx.z * gridDim.y) / yo.dims[2];
    const int idz = (blockIdx.y + blockIdx.z * gridDim.y) - idw * yo.dims[2];

    if (idx >= yo.dims[0] || idy >= yo.dims[1] || idz >= yo.dims[2] ||
        idw >= yo.dims[3])
        return;

    bool is_xo_off[] = {xo.dims[0] > 1, xo.dims[1] > 1, xo.dims[2] > 1,
                        xo.dims[3] > 1};
    bool is_yi_off[] = {true, true, true, true};
    is_yi_off[xdim]  = false;

    const int yo_idx =
        idw * yo.strides[3] + idz * yo.strides[2] + idy * yo.strides[1] + idx;
    int xo_idx = idx * is_xo_off[0];
    xo_idx += idw * xo.strides[3] * is_xo_off[3];
    xo_idx += idz * xo.strides[2] * is_xo_off[2];
    xo_idx += idy * xo.strides[1] * is_xo_off[1];

    const Tp x = (xo.ptr[xo_idx] - xi_beg) / xi_step;
    if (x < 0 || yi.dims[xdim] < x + 1) {
        yo.ptr[yo_idx] = scalar<Ty>(offGrid);
        return;
    }

    int yi_idx = idx * is_yi_off[0];
    yi_idx += idw * yi.strides[3] * is_yi_off[3];
    yi_idx += idz * yi.strides[2] * is_yi_off[2];
    yi_idx += idy * yi.strides[1] * is_yi_off[1];

    // FIXME: Only cubic interpolation is doing clamping
    // We need to make it consistent across all methods
    // Not changing the behavior because tests will fail
    bool clamp = order == 3;

    Interp1<Ty, Tp, order> interp;
    interp(yo, yo_idx, yi, yi_idx, x, method, 1, clamp, xdim);
}

}
