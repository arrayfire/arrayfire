/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

__kernel
void approx1_kernel(__global       Ty *d_yo, const KParam yo,
                    __global const Ty *d_yi,  const KParam yi,
                    __global const Tp *d_xo, const KParam xo, const int xdim,
                    const Tp xi_beg, const Tp xi_step,
                    const Ty offGrid, const int blocksMatX, const int batch, const int method)
{
    const int idw = get_group_id(1) / yo.dims[2];
    const int idz = get_group_id(1)  - idw * yo.dims[2];

    const int idy = get_group_id(0) / blocksMatX;
    const int blockIdx_x = get_group_id(0) - idy * blocksMatX;
    const int idx = get_local_id(0) + blockIdx_x * get_local_size(0);

    if(idx >= yo.dims[0] ||
       idy >= yo.dims[1] ||
       idz >= yo.dims[2] ||
       idw >= yo.dims[3])
        return;

    const int omId = idw * yo.strides[3] + idz * yo.strides[2]
        + idy * yo.strides[1] + idx + yo.offset;
    int xmid = idx + xo.offset;
    if(batch) xmid += idw * xo.strides[3] + idz * xo.strides[2] + idy * xo.strides[1];

    const Tp x = (d_xo[xmid] - xi_beg) / xi_step;
    if (x < 0 || yi.dims[0] < x+1) {
        d_yo[omId] = offGrid;
        return;
    }

    int ioff = idw * yi.strides[3] + idz * yi.strides[2] + idy * yi.strides[1] + yi.offset;

    // FIXME: Only cubic interpolation is doing clamping
    // We need to make it consistent across all methods
    // Not changing the behavior because tests will fail
    bool clamp = INTERP_ORDER == 3;

    interp1(d_yo, yo, omId,
            d_yi,  yi, ioff,
            x, method, 1, clamp);
}
