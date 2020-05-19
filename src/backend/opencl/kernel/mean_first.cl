/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

kernel void meanFirst(global To *oData, KParam oInfo,
#ifdef OUTPUT_WEIGHT
                      global Tw *owData, KParam owInfo,
#endif
                      const global Ti *iData, KParam iInfo,
#ifdef INPUT_WEIGHT
                      const global Tw *iwData, KParam iwInfo,
#endif
                      uint groups_x, uint groups_y, uint repeat) {
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

#ifdef INPUT_WEIGHT
    iwData += wid * iwInfo.strides[3] + zid * iwInfo.strides[2] +
              yid * iwInfo.strides[1] + iwInfo.offset;
#endif

    oData += wid * oInfo.strides[3] + zid * oInfo.strides[2] +
             yid * oInfo.strides[1] + oInfo.offset;

#ifdef OUTPUT_WEIGHT
    owData += wid * owInfo.strides[3] + zid * owInfo.strides[2] +
              yid * owInfo.strides[1] + owInfo.offset;
#endif

    bool cond =
        (yid < iInfo.dims[1]) && (zid < iInfo.dims[2]) && (wid < iInfo.dims[3]);

    local To s_val[THREADS_PER_GROUP];
    local Tw s_wt[THREADS_PER_GROUP];

    int last   = (xid + repeat * DIMX);
    int lim    = last > iInfo.dims[0] ? iInfo.dims[0] : last;
    To out_val = init_To;
    Tw out_wt  = init_Tw;

    if (cond && xid < lim) {
        out_val = transform(iData[xid]);
#ifdef INPUT_WEIGHT
        out_wt = iwData[xid];
#else
        out_wt = one_Tw;
#endif
    }

#ifdef INPUT_WEIGHT
    for (int id = xid + DIMX; cond && id < lim; id += DIMX) {
        binOp(&out_val, &out_wt, transform(iData[id]), iwData[id]);
    }
#else
    for (int id = xid + DIMX; cond && id < lim; id += DIMX) {
        binOp(&out_val, &out_wt, transform(iData[id]), one_Tw);
    }
#endif

    s_val[lid] = out_val;
    s_wt[lid]  = out_wt;
    barrier(CLK_LOCAL_MEM_FENCE);

    local To *s_vptr = s_val + lidy * DIMX;
    local Tw *s_wptr = s_wt + lidy * DIMX;

    if (DIMX == 256) {
        if (lidx < 128) {
            binOp(&out_val, &out_wt, s_vptr[lidx + 128], s_wptr[lidx + 128]);
            s_vptr[lidx] = out_val;
            s_wptr[lidx] = out_wt;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (DIMX >= 128) {
        if (lidx < 64) {
            binOp(&out_val, &out_wt, s_vptr[lidx + 64], s_wptr[lidx + 64]);
            s_vptr[lidx] = out_val;
            s_wptr[lidx] = out_wt;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (DIMX >= 64) {
        if (lidx < 32) {
            binOp(&out_val, &out_wt, s_vptr[lidx + 32], s_wptr[lidx + 32]);
            s_vptr[lidx] = out_val;
            s_wptr[lidx] = out_wt;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lidx < 16) {
        binOp(&out_val, &out_wt, s_vptr[lidx + 16], s_wptr[lidx + 16]);
        s_vptr[lidx] = out_val;
        s_wptr[lidx] = out_wt;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lidx < 8) {
        binOp(&out_val, &out_wt, s_vptr[lidx + 8], s_wptr[lidx + 8]);
        s_vptr[lidx] = out_val;
        s_wptr[lidx] = out_wt;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lidx < 4) {
        binOp(&out_val, &out_wt, s_vptr[lidx + 4], s_wptr[lidx + 4]);
        s_vptr[lidx] = out_val;
        s_wptr[lidx] = out_wt;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lidx < 2) {
        binOp(&out_val, &out_wt, s_vptr[lidx + 2], s_wptr[lidx + 2]);
        s_vptr[lidx] = out_val;
        s_wptr[lidx] = out_wt;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lidx < 1) {
        binOp(&out_val, &out_wt, s_vptr[lidx + 1], s_wptr[lidx + 1]);
        s_vptr[lidx] = out_val;
        s_wptr[lidx] = out_wt;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (cond && lidx == 0) {
        oData[groupId_x] = s_vptr[0];
#ifdef OUTPUT_WEIGHT
        owData[groupId_x] = s_wptr[0];
#endif
    }
}
