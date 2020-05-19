/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if CPLX
#define isZero(val) ((val.x == 0) && (val.y == 0))
#else
#define isZero(val) ((val == 0))
#endif

kernel void get_out_idx(global uint *oData, global uint *otData, KParam otInfo,
                        global uint *rtData, KParam rtInfo, global T *iData,
                        KParam iInfo, uint groups_x, uint groups_y, uint lim) {
    T Zero = ZERO;

    const uint lidx = get_local_id(0);
    const uint lidy = get_local_id(1);

    const uint zid       = get_group_id(0) / groups_x;
    const uint wid       = get_group_id(1) / groups_y;
    const uint groupId_x = get_group_id(0) - (groups_x)*zid;
    const uint groupId_y = get_group_id(1) - (groups_y)*wid;
    const uint xid       = groupId_x * get_local_size(0) * lim + lidx;
    const uint yid       = groupId_y * get_local_size(1) + lidy;

    const uint off = wid * otInfo.strides[3] + zid * otInfo.strides[2] +
                     yid * otInfo.strides[1];
    const uint gid = wid * rtInfo.strides[3] + zid * rtInfo.strides[2] +
                     yid * rtInfo.strides[1] + groupId_x;

    otData += wid * otInfo.strides[3] + zid * otInfo.strides[2] +
              yid * otInfo.strides[1];
    iData += wid * iInfo.strides[3] + zid * iInfo.strides[2] +
             yid * iInfo.strides[1] + iInfo.offset;

    bool cond = (yid < otInfo.dims[1]) && (zid < otInfo.dims[2]) &&
                (wid < otInfo.dims[3]);
    if (!cond) return;

    uint accum = (gid == 0) ? 0 : rtData[gid - 1];

    for (uint k = 0, id = xid; k < lim && id < otInfo.dims[0];
         k++, id += get_local_size(0)) {
        uint idx = otData[id] + accum;
        T ival   = iData[id];
        if (!isZero(ival)) oData[idx - 1] = (off + id);
    }
}
