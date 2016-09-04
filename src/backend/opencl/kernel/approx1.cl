/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

__kernel
void approx1_kernel(__global       Ty *d_out, const KParam out,
                    __global const Ty *d_in,  const KParam in,
                    __global const Tp *d_xpos, const KParam xpos,
                    const Ty offGrid, const int blocksMatX, const int batch)
{
    const int idw = get_group_id(1) / out.dims[2];
    const int idz = get_group_id(1)  - idw * out.dims[2];

    const int idy = get_group_id(0) / blocksMatX;
    const int blockIdx_x = get_group_id(0) - idy * blocksMatX;
    const int idx = get_local_id(0) + blockIdx_x * get_local_size(0);

    if(idx >= out.dims[0] ||
       idy >= out.dims[1] ||
       idz >= out.dims[2] ||
       idw >= out.dims[3])
        return;

    const int omId = idw * out.strides[3] + idz * out.strides[2]
        + idy * out.strides[1] + idx + out.offset;
    int xmid = idx + xpos.offset;
    if(batch) xmid += idw * xpos.strides[3] + idz * xpos.strides[2] + idy * xpos.strides[1];

    const Tp x = d_xpos[xmid];
    if (x < 0 || in.dims[0] < x+1) {
        d_out[omId] = offGrid;
        return;
    }

    int ioff = idw * in.strides[3] + idz * in.strides[2] + idy * in.strides[1] + in.offset;
    d_out[omId] = interp1(d_in, in, ioff, x);
}
