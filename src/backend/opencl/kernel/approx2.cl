/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

__kernel void approx2_kernel(__global Ty *d_out, const KParam out,
                             __global const Ty *d_in, const KParam in,
                             __global const Tp *d_xpos, const KParam xpos,
                             __global const Tp *d_ypos, const KParam ypos,
                             const Ty offGrid, const int blocksMatX,
                             const int blocksMatY, const int batch,
                             int method) {
    const int idz = get_group_id(0) / blocksMatX;
    const int idw = get_group_id(1) / blocksMatY;

    const int blockIdx_x = get_group_id(0) - idz * blocksMatX;
    const int blockIdx_y = get_group_id(1) - idw * blocksMatY;

    const int idx = get_local_id(0) + blockIdx_x * get_local_size(0);
    const int idy = get_local_id(1) + blockIdx_y * get_local_size(1);

    if (idx >= out.dims[0] || idy >= out.dims[1] || idz >= out.dims[2] ||
        idw >= out.dims[3])
        return;

    const int omId = idw * out.strides[3] + idz * out.strides[2] +
                     idy * out.strides[1] + idx + out.offset;
    int xmid = idy * xpos.strides[1] + idx + xpos.offset;
    int ymid = idy * ypos.strides[1] + idx + ypos.offset;
    if (batch) {
        xmid += idw * xpos.strides[3] + idz * xpos.strides[2];
        ymid += idw * ypos.strides[3] + idz * ypos.strides[2];
    }

    const Tp x = d_xpos[xmid], y = d_ypos[ymid];
    if (x < 0 || y < 0 || in.dims[0] < x + 1 || in.dims[1] < y + 1) {
        d_out[omId] = offGrid;
        return;
    }

    int ioff = idw * in.strides[3] + idz * in.strides[2] + in.offset;

    // FIXME: Only cubic interpolation is doing clamping
    // We need to make it consistent across all methods
    // Not changing the behavior because tests will fail
    bool clamp = INTERP_ORDER == 3;

    interp2(d_out, out, omId, d_in, in, ioff, x, y, method, 1, clamp);
}
