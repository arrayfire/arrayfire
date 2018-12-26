/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

__kernel void diagCreateKernel(__global T *oData, KParam oInfo,
                               const __global T *iData, KParam iInfo, int num,
                               int groups_x) {
    unsigned idz       = get_group_id(0) / groups_x;
    unsigned groupId_x = get_group_id(0) - idz * groups_x;

    unsigned idx = get_local_id(0) + groupId_x * get_local_size(0);
    unsigned idy = get_global_id(1);

    if (idx >= oInfo.dims[0] || idy >= oInfo.dims[1] || idz >= oInfo.dims[2])
        return;

    __global T *optr =
        oData + idz * oInfo.strides[2] + idy * oInfo.strides[1] + idx;
    const __global T *iptr =
        iData + idz * iInfo.strides[1] + ((num > 0) ? idx : idy) + iInfo.offset;

    T val = (idx == (idy - num)) ? *iptr : ZERO;
    *optr = val;
}
