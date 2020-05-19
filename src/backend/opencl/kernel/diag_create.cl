/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

kernel void diagCreateKernel(global T *oData, KParam oInfo,
                               const global T *iData, KParam iInfo, int num,
                               int groups_x) {
    unsigned idz       = get_group_id(0) / groups_x;
    unsigned groupId_x = get_group_id(0) - idz * groups_x;

    unsigned idx = get_local_id(0) + groupId_x * get_local_size(0);
    unsigned idy = get_global_id(1);

    if (idx >= oInfo.dims[0] || idy >= oInfo.dims[1] || idz >= oInfo.dims[2])
        return;

    global T *optr =
        oData + idz * oInfo.strides[2] + idy * oInfo.strides[1] + idx;
    const global T *iptr =
        iData + idz * iInfo.strides[1] + ((num > 0) ? idx : idy) + iInfo.offset;

    T val = (idx == (idy - num)) ? *iptr : (T)(ZERO);
    *optr = val;
}
