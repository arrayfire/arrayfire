/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

__kernel void approx1_kernel(__global Ty *d_yo, const KParam yo,
                             __global const Ty *d_yi, const KParam yi,
                             __global const Tp *d_xo, const KParam xo,
                             const int xdim, const Tp xi_beg, const Tp xi_step,
                             const Ty offGrid, const int blocksMatX,
                             const int batch, const int method) {
    const int idw = get_group_id(1) / yo.dims[2];
    const int idz = get_group_id(1) - idw * yo.dims[2];

    const int idy        = get_group_id(0) / blocksMatX;
    const int blockIdx_x = get_group_id(0) - idy * blocksMatX;
    const int idx        = get_local_id(0) + blockIdx_x * get_local_size(0);

    if (idx >= yo.dims[0] || idy >= yo.dims[1] || idz >= yo.dims[2] ||
        idw >= yo.dims[3])
        return;

    bool is_xo_off[] = {xo.dims[0] > 1, xo.dims[1] > 1, xo.dims[2] > 1,
                        xo.dims[3] > 1};
    bool is_yi_off[] = {true, true, true, true};
    is_yi_off[xdim]  = false;

    const int yo_idx = idw * yo.strides[3] + idz * yo.strides[2] +
                       idy * yo.strides[1] + idx + yo.offset;

    int xo_idx = idx * is_xo_off[0] + xo.offset;
    xo_idx += idw * xo.strides[3] * is_xo_off[3];
    xo_idx += idz * xo.strides[2] * is_xo_off[2];
    xo_idx += idy * xo.strides[1] * is_xo_off[1];

    const Tp x = (d_xo[xo_idx] - xi_beg) / xi_step;
    if (x < 0 || yi.dims[xdim] < x + 1) {
        d_yo[yo_idx] = offGrid;
        return;
    }

    int yi_idx = idx * is_yi_off[0] + yi.offset;
    yi_idx += idw * yi.strides[3] * is_yi_off[3];
    yi_idx += idz * yi.strides[2] * is_yi_off[2];
    yi_idx += idy * yi.strides[1] * is_yi_off[1];

    // FIXME: Only cubic interpolation is doing clamping
    // We need to make it consistent across all methods
    // Not changing the behavior because tests will fail
    bool clamp = INTERP_ORDER == 3;

    interp1_dim(d_yo, yo, yo_idx, d_yi, yi, yi_idx, x, method, 1, clamp, xdim);
}
