/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

__kernel
void reduce_dim_kernel(__global To *oData,
                       KParam oInfo,
                       const __global Ti *iData,
                       KParam iInfo,
                       uint groups_x, uint groups_y, uint group_dim,
                       int change_nan, To nanval)
{
    const uint lidx = get_local_id(0);
    const uint lidy = get_local_id(1);
    const uint lid  = lidy * THREADS_X + lidx;

    const uint zid = get_group_id(0) / groups_x;
    const uint wid = get_group_id(1) / groups_y;
    const uint groupId_x = get_group_id(0) - (groups_x) * zid;
    const uint groupId_y = get_group_id(1) - (groups_y) * wid;
    const uint xid = groupId_x * get_local_size(0) + lidx;
    const uint yid = groupId_y;

    uint ids[4] = {xid, yid, zid, wid};

    // There is only one element per group for out
    // There are get_local_size(1) elements per group for in
    // Hence increment ids[dim] just after offseting out and before offsetting in
    oData += ids[3] * oInfo.strides[3] + ids[2] * oInfo.strides[2] +
        ids[1] * oInfo.strides[1] + ids[0] + oInfo.offset;
    const uint id_dim_out = ids[dim];

    ids[dim] = ids[dim] * get_local_size(1) + lidy;
    iData  += ids[3] * iInfo.strides[3] + ids[2] * iInfo.strides[2] +
        ids[1] * iInfo.strides[1] + ids[0] + iInfo.offset;
    const uint id_dim_in = ids[dim];

    const uint istride_dim = iInfo.strides[dim];

    bool is_valid =
        (ids[0] < iInfo.dims[0]) &&
        (ids[1] < iInfo.dims[1]) &&
        (ids[2] < iInfo.dims[2]) &&
        (ids[3] < iInfo.dims[3]);

    __local To s_val[THREADS_X * DIMY];

    To out_val = init;
    for (int id = id_dim_in; is_valid && (id < iInfo.dims[dim]);
         id += group_dim * get_local_size(1)) {

        To in_val = transform(*iData);
        if (change_nan) in_val = !IS_NAN(in_val) ? in_val : nanval;
        out_val = binOp(in_val, out_val);
        iData = iData + group_dim * get_local_size(1) * istride_dim;
    }

    s_val[lid] = out_val;

    __local To *s_ptr = s_val + lid;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (DIMY == 8) {
        if (lidy < 4) *s_ptr = binOp(*s_ptr, s_ptr[THREADS_X * 4]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (DIMY >= 4) {
        if (lidy < 2) *s_ptr = binOp(*s_ptr, s_ptr[THREADS_X * 2]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (DIMY >= 2) {
        if (lidy < 1) *s_ptr = binOp(*s_ptr, s_ptr[THREADS_X * 1]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lidy == 0 && is_valid &&
        (id_dim_out < oInfo.dims[dim])) {
        *oData = *s_ptr;
    }

}
