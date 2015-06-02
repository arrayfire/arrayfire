/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

__kernel
void identity_kernel(__global T *oData, KParam oInfo, int groups_x, int groups_y)
{

    unsigned idz = get_group_id(0) / groups_x;
    unsigned idw = get_group_id(1) / groups_y;

    unsigned groupId_x = get_group_id(0) - idz * groups_x;
    unsigned groupId_y = get_group_id(1) - idw * groups_y;

    unsigned idx = get_local_id(0) + groupId_x * get_local_size(0);
    unsigned idy = get_local_id(1) + groupId_y * get_local_size(1);

    if(idx >= oInfo.dims[0] ||
       idy >= oInfo.dims[1] ||
       idz >= oInfo.dims[2] ||
       idw >= oInfo.dims[3])
        return;

    __global T *ptr = oData + idz * oInfo.strides[2] + idw * oInfo.strides[3];
    T val = (idx == idy) ? ONE : ZERO;
    ptr[idx + idy * oInfo.strides[1]] = val;
}
