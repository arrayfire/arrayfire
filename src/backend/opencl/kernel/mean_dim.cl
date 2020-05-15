/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

kernel void meanDim(global To *oData, KParam oInfo,
#ifdef OUTPUT_WEIGHT
                    global Tw *owData, KParam owInfo,
#endif
                    const global Ti *iData, KParam iInfo,
#ifdef INPUT_WEIGHT
                    const global Tw *iwData, KParam iwInfo,
#endif
                    uint groups_x, uint groups_y, uint group_dim) {
    const uint lidx = get_local_id(0);
    const uint lidy = get_local_id(1);
    const uint lid  = lidy * THREADS_X + lidx;

    const uint zid       = get_group_id(0) / groups_x;
    const uint wid       = get_group_id(1) / groups_y;
    const uint groupId_x = get_group_id(0) - (groups_x)*zid;
    const uint groupId_y = get_group_id(1) - (groups_y)*wid;
    const uint xid       = groupId_x * get_local_size(0) + lidx;
    const uint yid       = groupId_y;

    uint ids[4] = {xid, yid, zid, wid};

    // There is only one element per group for out
    // There are get_local_size(1) elements per group for in
    // Hence increment ids[kDim] just after offseting out and before offsetting
    // in
    oData += ids[3] * oInfo.strides[3] + ids[2] * oInfo.strides[2] +
             ids[1] * oInfo.strides[1] + ids[0] + oInfo.offset;

#ifdef OUTPUT_WEIGHT
    owData += ids[3] * oInfo.strides[3] + ids[2] * oInfo.strides[2] +
              ids[1] * oInfo.strides[1] + ids[0] + oInfo.offset;
#endif
    const uint id_dim_out = ids[kDim];

    ids[kDim] = ids[kDim] * get_local_size(1) + lidy;

    iData += ids[3] * iInfo.strides[3] + ids[2] * iInfo.strides[2] +
             ids[1] * iInfo.strides[1] + ids[0] + iInfo.offset;

#ifdef INPUT_WEIGHT
    iwData += ids[3] * iInfo.strides[3] + ids[2] * iInfo.strides[2] +
              ids[1] * iInfo.strides[1] + ids[0] + iInfo.offset;
#endif

    const uint id_dim_in   = ids[kDim];
    const uint istride_dim = iInfo.strides[kDim];

    bool is_valid = (ids[0] < iInfo.dims[0]) && (ids[1] < iInfo.dims[1]) &&
                    (ids[2] < iInfo.dims[2]) && (ids[3] < iInfo.dims[3]);

    local To s_val[THREADS_X * DIMY];
    local Tw s_wt[THREADS_X * DIMY];

    To out_val = init_To;
    Tw out_wt  = init_Tw;

    if (is_valid && id_dim_in < iInfo.dims[kDim]) {
        out_val = transform(*iData);
#ifdef INPUT_WEIGHT
        out_wt = *iwData;
#else
        out_wt = one_Tw;
#endif
    }

    const uint id_dim_in_start = id_dim_in + group_dim * get_local_size(1);

#ifdef INPUT_WEIGHT
    for (int id = id_dim_in_start; is_valid && (id < iInfo.dims[kDim]);
         id += group_dim * get_local_size(1)) {
        iData  = iData + group_dim * get_local_size(1) * istride_dim;
        iwData = iwData + group_dim * get_local_size(1) * istride_dim;
        binOp(&out_val, &out_wt, transform(*iData), *iwData);
    }
#else
    for (int id = id_dim_in_start; is_valid && (id < iInfo.dims[kDim]);
         id += group_dim * get_local_size(1)) {
        iData = iData + group_dim * get_local_size(1) * istride_dim;
        binOp(&out_val, &out_wt, transform(*iData), one_Tw);
    }
#endif

    s_val[lid] = out_val;
    s_wt[lid]  = out_wt;

    local To *s_vptr = s_val + lid;
    local Tw *s_wptr = s_wt + lid;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (DIMY == 8) {
        if (lidy < 4) {
            binOp(&out_val, &out_wt, s_vptr[THREADS_X * 4],
                  s_wptr[THREADS_X * 4]);
            *s_vptr = out_val;
            *s_wptr = out_wt;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (DIMY >= 4) {
        if (lidy < 2) {
            binOp(&out_val, &out_wt, s_vptr[THREADS_X * 2],
                  s_wptr[THREADS_X * 2]);
            *s_vptr = out_val;
            *s_wptr = out_wt;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (DIMY >= 2) {
        if (lidy < 1) {
            binOp(&out_val, &out_wt, s_vptr[THREADS_X * 1],
                  s_wptr[THREADS_X * 1]);
            *s_vptr = out_val;
            *s_wptr = out_wt;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lidy == 0 && is_valid && (id_dim_out < oInfo.dims[kDim])) {
        *oData = *s_vptr;
#ifdef OUTPUT_WEIGHT
        *owData = *s_wptr;
#endif
    }
}
