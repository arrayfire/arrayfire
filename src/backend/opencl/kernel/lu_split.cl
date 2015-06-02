/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

__kernel
void lu_split_kernel(__global T *lptr, KParam linfo,
                     __global T *uptr, KParam uinfo,
                     const __global T *iptr, KParam iinfo,
                     const int groups_x, const int groups_y)
{
    const int oz = get_group_id(0) / groups_x;
    const int ow = get_group_id(1) / groups_y;

    const int groupIdx_0 = get_group_id(0) - oz * groups_x;
    const int groupIdx_1 = get_group_id(1) - ow * groups_y;

    const int xx = get_local_id(0) + groupIdx_0 * get_local_size(0);
    const int yy = get_local_id(1) + groupIdx_1 * get_local_size(1);

    const int incy = groups_y * get_local_size(1);
    const int incx = groups_x * get_local_size(0);

    __global T *d_l = lptr;
    __global T *d_u = uptr;
    __global T *d_i = iptr;

    if(oz < iinfo.dims[2] && ow < iinfo.dims[3]) {
        d_i = d_i + oz * iinfo.strides[2]    + ow * iinfo.strides[3];
        d_l = d_l + oz * linfo.strides[2] + ow * linfo.strides[3];
        d_u = d_u + oz * uinfo.strides[2] + ow * uinfo.strides[3];

        for (int oy = yy; oy < iinfo.dims[1]; oy += incy) {
            __global T *Yd_i = d_i + oy * iinfo.strides[1];
            __global T *Yd_l = d_l +  oy * linfo.strides[1];
            __global T *Yd_u = d_u +  oy * uinfo.strides[1];
            for (int ox = xx; ox < iinfo.dims[0]; ox += incx) {
                if(ox > oy) {
                    if(same_dims || oy < linfo.dims[1])
                        Yd_l[ox] = Yd_i[ox];
                    if(!same_dims || ox < uinfo.dims[0])
                        Yd_u[ox] = ZERO;
                } else if (oy > ox) {
                    if(same_dims || oy < linfo.dims[1])
                        Yd_l[ox] = ZERO;
                    if(!same_dims || ox < uinfo.dims[0])
                        Yd_u[ox] = Yd_i[ox];
                } else if(ox == oy) {
                    if(same_dims || oy < linfo.dims[1])
                        Yd_l[ox] = ONE;
                    if(!same_dims || ox < uinfo.dims[0])
                        Yd_u[ox] = Yd_i[ox];
                }
            }
        }
    }
}
