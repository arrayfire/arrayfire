/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

__kernel
void approx2_kernel(__global       Ty *d_zo, const KParam zo,
                    __global const Ty *d_zi,  const KParam zi,
                    __global const Tp *d_xo, const KParam xo, const int xdim,
                    __global const Tp *d_yo, const KParam yo, const int ydim,
                    const Tp xi_beg, const Tp xi_step,
                    const Tp yi_beg, const Tp yi_step,
                    const Ty offGrid, const int blocksMatX, const int blocksMatY,
                    const int batch, int method)
{
    const int idz = get_group_id(0) / blocksMatX;
    const int idw = get_group_id(1) / blocksMatY;

    const int blockIdx_x = get_group_id(0) - idz * blocksMatX;
    const int blockIdx_y = get_group_id(1) - idw * blocksMatY;

    const int idx = get_local_id(0) + blockIdx_x * get_local_size(0);
    const int idy = get_local_id(1) + blockIdx_y * get_local_size(1);

    if(idx >= zo.dims[0] ||
       idy >= zo.dims[1] ||
       idz >= zo.dims[2] ||
       idw >= zo.dims[3])
        return;

    const int omId = idw * zo.strides[3] + idz * zo.strides[2]
        + idy * zo.strides[1] + idx + zo.offset;
    int xmid = idy * xo.strides[1] + idx + xo.offset;
    int ymid = idy * yo.strides[1] + idx + yo.offset;
    if(batch) {
        xmid += idw * xo.strides[3] + idz * xo.strides[2];
        ymid += idw * yo.strides[3] + idz * yo.strides[2];
    }

    const Tp x = (d_xo[xmid] - xi_beg) / xi_step;
    const Tp y = (d_yo[ymid] - yi_beg) / yi_step;
    if (x < 0 || y < 0 || zi.dims[0] < x+1 || zi.dims[1] < y+1) {
        d_zo[omId] = offGrid;
        return;
    }

    int ioff = idw * zi.strides[3] + idz * zi.strides[2] + zi.offset;

    // FIXME: Only cubic interpolation is doing clamping
    // We need to make it consistent across all methods
    // Not changing the behavior because tests will fail
    bool clamp = INTERP_ORDER == 3;

    interp2(d_zo, zo, omId,
            d_zi,  zi, ioff,
            x, y, method, 1, clamp);
}
