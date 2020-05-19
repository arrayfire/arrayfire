/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

kernel void diagExtractKernel(global T *oData, KParam oInfo,
                                const global T *iData, KParam iInfo, int num,
                                int groups_z) {
    unsigned idw = get_group_id(1) / groups_z;
    unsigned idz = get_group_id(1) - idw * groups_z;

    unsigned idx = get_global_id(0);

    if (idx >= oInfo.dims[0] || idz >= oInfo.dims[2] || idw >= oInfo.dims[3])
        return;

    global T *optr =
        oData + idz * oInfo.strides[2] + idw * oInfo.strides[3] + idx;

    if (idx >= iInfo.dims[0] || idx >= iInfo.dims[1]) {
        *optr = (T)(ZERO);
        return;
    }

    int i_off =
        (num > 0) ? (num * iInfo.strides[1] + idx) : (idx - num) + iInfo.offset;

    const global T *iptr =
        iData + idz * iInfo.strides[2] + idw * iInfo.strides[3] + i_off;

    *optr = iptr[idx * iInfo.strides[1]];
}
