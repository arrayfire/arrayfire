/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

__kernel
void triangle_kernel(__global T *rptr, KParam rinfo,
                     const __global T *iptr, KParam iinfo,
                     const dim_type groups_x, const dim_type groups_y)
{
    const dim_type oz = get_group_id(0) / groups_x;
    const dim_type ow = get_group_id(1) / groups_y;

    const dim_type groupId_0 = get_group_id(0) - oz * groups_x;
    const dim_type groupId_1 = get_group_id(1) - ow * groups_y;

    const dim_type xx = get_local_id(0) + groupId_0 * get_local_size(0);
    const dim_type yy = get_local_id(1) + groupId_1 * get_local_size(1);

    const dim_type incy = groups_y * get_local_size(1);
    const dim_type incx = groups_x * get_local_size(0);

    __global T *d_r = rptr;
    const __global T *d_i = iptr;

    if(oz < rinfo.dims[2] && ow < rinfo.dims[3]) {
        d_i = d_i + oz * iinfo.strides[2] + ow * iinfo.strides[3];
        d_r = d_r + oz * rinfo.strides[2] + ow * rinfo.strides[3];

        for (dim_type oy = yy; oy < rinfo.dims[1]; oy += incy) {
            const __global T *Yd_i = d_i + oy * iinfo.strides[1];
            __global T *Yd_r = d_r +  oy * rinfo.strides[1];

            for (dim_type ox = xx; ox < rinfo.dims[0]; ox += incx) {

                bool cond = is_upper ? (oy >= ox) : (oy <= ox);
                if(cond) {
                    Yd_r[ox] = Yd_i[ox];
                } else {
                    Yd_r[ox] = ZERO;
                }
            }
        }
    }
}
