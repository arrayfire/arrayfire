/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

__kernel
void reduce_first_kernel(__global To *oData,
                         KParam oInfo,
                         const __global Ti *iData,
                         KParam iInfo,
                         uint groups_x, uint groups_y, uint repeat,
                         int change_nan, To nanval)
{
    const uint lidx = get_local_id(0);
    const uint lidy = get_local_id(1);
    const uint lid  = lidy * get_local_size(0) + lidx;

    const uint zid = get_group_id(0) / groups_x;
    const uint wid = get_group_id(1) / groups_y;
    const uint groupId_x = get_group_id(0) - (groups_x) * zid;
    const uint groupId_y = get_group_id(1) - (groups_y) * wid;
    const uint xid = groupId_x * get_local_size(0) * repeat + lidx;
    const uint yid = groupId_y * get_local_size(1) + lidy;

    iData += wid * iInfo.strides[3] + zid * iInfo.strides[2] +
        yid * iInfo.strides[1] + iInfo.offset;
    oData += wid * oInfo.strides[3] + zid * oInfo.strides[2] +
        yid * oInfo.strides[1] + oInfo.offset;

    bool cond = (yid < iInfo.dims[1]) && (zid < iInfo.dims[2]) && (wid < iInfo.dims[3]);

    __local To s_val[THREADS_PER_GROUP];

    int last = (xid + repeat * DIMX);
    int lim = last > iInfo.dims[0] ? iInfo.dims[0] : last;
    To out_val = init;

    for (int id = xid; cond && id < lim; id += DIMX) {
        To in_val = transform(iData[id]);
        if (change_nan) in_val = !IS_NAN(in_val) ? in_val : nanval;
        out_val = binOp(in_val, out_val);
    }

    s_val[lid] = out_val;
    barrier(CLK_LOCAL_MEM_FENCE);
    __local To *s_ptr = s_val + lidy * DIMX;

    if (DIMX == 256) {
        if (lidx < 128) s_ptr[lidx] = binOp(s_ptr[lidx], s_ptr[lidx + 128]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (DIMX >= 128) {
        if (lidx <  64) s_ptr[lidx] = binOp(s_ptr[lidx], s_ptr[lidx +  64]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (DIMX >=  64) {
        if (lidx <  32) s_ptr[lidx] = binOp(s_ptr[lidx], s_ptr[lidx +  32]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lidx < 16) s_ptr[lidx] = binOp(s_ptr[lidx], s_ptr[lidx + 16]);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lidx <  8) s_ptr[lidx] = binOp(s_ptr[lidx], s_ptr[lidx +  8]);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lidx <  4) s_ptr[lidx] = binOp(s_ptr[lidx], s_ptr[lidx +  4]);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lidx <  2) s_ptr[lidx] = binOp(s_ptr[lidx], s_ptr[lidx +  2]);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lidx <  1) s_ptr[lidx] = binOp(s_ptr[lidx], s_ptr[lidx +  1]);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (cond && lidx == 0) {
        oData[groupId_x] = s_ptr[0];
    }
}
