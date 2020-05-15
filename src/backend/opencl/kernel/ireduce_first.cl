/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

kernel void ireduce_first_kernel(global T *oData, KParam oInfo,
                                   global uint *olData,
                                   const global T *iData, KParam iInfo,
                                   const global uint *ilData, uint groups_x,
                                   uint groups_y, uint repeat,
                                   global uint *rlenptr, KParam rlen) {
    const uint lidx = get_local_id(0);
    const uint lidy = get_local_id(1);
    const uint lid  = lidy * get_local_size(0) + lidx;

    const uint zid       = get_group_id(0) / groups_x;
    const uint wid       = get_group_id(1) / groups_y;
    const uint groupId_x = get_group_id(0) - (groups_x)*zid;
    const uint groupId_y = get_group_id(1) - (groups_y)*wid;
    const uint xid       = groupId_x * get_local_size(0) * repeat + lidx;
    const uint yid       = groupId_y * get_local_size(1) + lidy;

    iData += wid * iInfo.strides[3] + zid * iInfo.strides[2] +
             yid * iInfo.strides[1] + iInfo.offset;

    if (!IS_FIRST) {
        ilData += wid * iInfo.strides[3] + zid * iInfo.strides[2] +
                  yid * iInfo.strides[1] + iInfo.offset;
    }

    oData += wid * oInfo.strides[3] + zid * oInfo.strides[2] +
             yid * oInfo.strides[1] + oInfo.offset;

    olData += wid * oInfo.strides[3] + zid * oInfo.strides[2] +
              yid * oInfo.strides[1] + oInfo.offset;

    rlenptr += (rlenptr) ? wid * rlen.strides[3] + zid * rlen.strides[2] +
                               yid * rlen.strides[1] + rlen.offset
                         : 0;

    bool cond =
        (yid < iInfo.dims[1]) && (zid < iInfo.dims[2]) && (wid < iInfo.dims[3]);

    local T s_val[THREADS_PER_GROUP];
    local uint s_idx[THREADS_PER_GROUP];

    int last = (xid + repeat * DIMX);

    int minlen = rlenptr ? min(*rlenptr, (uint)iInfo.dims[0]) : iInfo.dims[0];

    int lim      = last > minlen ? minlen : last;
    T out_val    = init;
    uint out_idx = xid;

    if (cond && xid < lim && !is_nan(iData[xid])) {
        out_val = iData[xid];
        if (!IS_FIRST) out_idx = ilData[xid];
    }

    for (int id = xid + DIMX; cond && id < lim; id += DIMX) {
#if IS_FIRST
        binOp(&out_val, &out_idx, iData[id], id);
#else
        binOp(&out_val, &out_idx, iData[id], ilData[id]);
#endif
    }

    s_val[lid] = out_val;
    s_idx[lid] = out_idx;
    barrier(CLK_LOCAL_MEM_FENCE);

    local T *s_vptr    = s_val + lidy * DIMX;
    local uint *s_iptr = s_idx + lidy * DIMX;

    if (DIMX == 256) {
        if (lidx < 128) {
            binOp(&out_val, &out_idx, s_vptr[lidx + 128], s_iptr[lidx + 128]);
            s_vptr[lidx] = out_val;
            s_iptr[lidx] = out_idx;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (DIMX >= 128) {
        if (lidx < 64) {
            binOp(&out_val, &out_idx, s_vptr[lidx + 64], s_iptr[lidx + 64]);
            s_vptr[lidx] = out_val;
            s_iptr[lidx] = out_idx;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (DIMX >= 64) {
        if (lidx < 32) {
            binOp(&out_val, &out_idx, s_vptr[lidx + 32], s_iptr[lidx + 32]);
            s_vptr[lidx] = out_val;
            s_iptr[lidx] = out_idx;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lidx < 16) {
        binOp(&out_val, &out_idx, s_vptr[lidx + 16], s_iptr[lidx + 16]);
        s_vptr[lidx] = out_val;
        s_iptr[lidx] = out_idx;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lidx < 8) {
        binOp(&out_val, &out_idx, s_vptr[lidx + 8], s_iptr[lidx + 8]);
        s_vptr[lidx] = out_val;
        s_iptr[lidx] = out_idx;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lidx < 4) {
        binOp(&out_val, &out_idx, s_vptr[lidx + 4], s_iptr[lidx + 4]);
        s_vptr[lidx] = out_val;
        s_iptr[lidx] = out_idx;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lidx < 2) {
        binOp(&out_val, &out_idx, s_vptr[lidx + 2], s_iptr[lidx + 2]);
        s_vptr[lidx] = out_val;
        s_iptr[lidx] = out_idx;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lidx < 1) {
        binOp(&out_val, &out_idx, s_vptr[lidx + 1], s_iptr[lidx + 1]);
        s_vptr[lidx] = out_val;
        s_iptr[lidx] = out_idx;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (cond && lidx == 0) {
        oData[groupId_x]  = s_vptr[0];
        olData[groupId_x] = s_iptr[0];
    }
}
