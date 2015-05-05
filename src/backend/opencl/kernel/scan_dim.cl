/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

__kernel
void scan_dim_kernel(__global To *oData, KParam oInfo,
                     __global To *tData, KParam tInfo,
                     const __global Ti *iData, KParam iInfo,
                     uint groups_x,
                     uint groups_y,
                     uint groups_dim,
                     uint lim)
{
    const int lidx = get_local_id(0);
    const int lidy = get_local_id(1);
    const int lid  = lidy * THREADS_X + lidx;

    const int zid = get_group_id(0) / groups_x;
    const int wid = get_group_id(1) / groups_y;
    const int groupId_x = get_group_id(0) - (groups_x) * zid;
    const int groupId_y = get_group_id(1) - (groups_y) * wid;
    const int xid = groupId_x * get_local_size(0) + lidx;
    const int yid = groupId_y;

    int ids[4] = {xid, yid, zid, wid};

    // There is only one element per group for out
    // There are DIMY elements per group for in
    // Hence increment ids[dim] just after offseting out and before offsetting in
    tData += ids[3] * tInfo.strides[3] + ids[2] * tInfo.strides[2] + ids[1] * tInfo.strides[1] + ids[0];
    const int groupId_dim = ids[dim];

    ids[dim] = ids[dim] * DIMY * lim + lidy;
    oData  += ids[3] * oInfo.strides[3] + ids[2] * oInfo.strides[2] + ids[1] * oInfo.strides[1] + ids[0];
    iData  += ids[3] *  iInfo.strides[3] + ids[2] *  iInfo.strides[2] + ids[1] *  iInfo.strides[1] + ids[0];
    int id_dim = ids[dim];
    const int out_dim = oInfo.dims[dim];

    bool is_valid =
        (ids[0] < oInfo.dims[0]) &&
        (ids[1] < oInfo.dims[1]) &&
        (ids[2] < oInfo.dims[2]) &&
        (ids[3] < oInfo.dims[3]);

    const int ostride_dim = oInfo.strides[dim];
    const int istride_dim =  iInfo.strides[dim];

    __local To l_val0[THREADS_X * DIMY];
    __local To l_val1[THREADS_X * DIMY];
    __local To *l_val = l_val0;
    __local To l_tmp[THREADS_X];

    bool flip = 0;
    const To init_val  = init;
    To val = init_val;
    const bool isLast = (lidy == (DIMY - 1));

    for (int k = 0; k < lim; k++) {

        if (isLast) l_tmp[lidx] = val;

        bool cond = (is_valid) && (id_dim < out_dim);
        val = cond ? transform(*iData) : init_val;
        l_val[lid] = val;
        barrier(CLK_LOCAL_MEM_FENCE);

        int start = 0;
        for (int off = 1; off < DIMY; off *= 2) {

            if (lidy >= off) val = binOp(val, l_val[lid - off * THREADS_X]);

            flip = 1 - flip;
            l_val = flip ? l_val1 : l_val0;
            l_val[lid] = val;

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        val = binOp(val, l_tmp[lidx]);
        if (cond) *oData = val;
        barrier(CLK_LOCAL_MEM_FENCE);

        id_dim += DIMY;
        iData += DIMY * istride_dim;
        oData += DIMY * ostride_dim;
    }

    if (!isFinalPass &&
        is_valid &&
        (groupId_dim < tInfo.dims[dim]) &&
        isLast) {
        *tData = val;
    }
}

__kernel
void bcast_dim_kernel(__global To *oData, KParam oInfo,
                      const __global To *tData, KParam tInfo,
                      uint groups_x,
                      uint groups_y,
                      uint groups_dim,
                      uint lim)
{
    const int lidx = get_local_id(0);
    const int lidy = get_local_id(1);
    const int lid  = lidy * THREADS_X + lidx;

    const int zid = get_group_id(0) / groups_x;
    const int wid = get_group_id(1) / groups_y;
    const int groupId_x = get_group_id(0) - (groups_x) * zid;
    const int groupId_y = get_group_id(1) - (groups_y) * wid;
    const int xid = groupId_x * get_local_size(0) + lidx;
    const int yid = groupId_y;

    int ids[4] = {xid, yid, zid, wid};
    const int groupId_dim = ids[dim];

    if (groupId_dim != 0) {

        // There is only one element per group for out
        // There are DIMY elements per group for in
        // Hence increment ids[dim] just after offseting out and before offsetting in
        tData += ids[3] * tInfo.strides[3] + ids[2] * tInfo.strides[2] + ids[1] * tInfo.strides[1] + ids[0];

        ids[dim] = ids[dim] * DIMY * lim + lidy;
        oData  += ids[3] * oInfo.strides[3] + ids[2] * oInfo.strides[2] + ids[1] * oInfo.strides[1] + ids[0];

        const int id_dim = ids[dim];
        const int out_dim = oInfo.dims[dim];

        bool is_valid =
            (ids[0] < oInfo.dims[0]) &&
            (ids[1] < oInfo.dims[1]) &&
            (ids[2] < oInfo.dims[2]) &&
            (ids[3] < oInfo.dims[3]);

        if (is_valid) {

            To accum = *(tData - tInfo.strides[dim]);

            const int ostride_dim = oInfo.strides[dim];

            for (int k = 0, id = id_dim;
                 is_valid && k < lim && (id < out_dim);
                 k++, id += DIMY) {

                *oData = binOp(*oData, accum);
                oData += DIMY * ostride_dim;
            }
        }
    }
}
