/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

kernel void ireduce_dim_kernel(global T *oData, KParam oInfo,
                                 global uint *olData, const __global T *iData,
                                 KParam iInfo, const global uint *ilData,
                                 uint groups_x, uint groups_y, uint group_dim,
                                 global uint *rlenptr, KParam rlen) {
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
    bool rlen_valid = (ids[0] < rlen.dims[0]) && (ids[1] < rlen.dims[1]) &&
                      (ids[2] < rlen.dims[2]) && (ids[3] < rlen.dims[3]);
    rlenptr += (rlenptr && rlen_valid)
                   ? ids[3] * rlen.strides[3] + ids[2] * rlen.strides[2] +
                         ids[1] * rlen.strides[1] + ids[0] + rlen.offset
                   : 0;

    oData += ids[3] * oInfo.strides[3] + ids[2] * oInfo.strides[2] +
             ids[1] * oInfo.strides[1] + ids[0] + oInfo.offset;
    olData += ids[3] * oInfo.strides[3] + ids[2] * oInfo.strides[2] +
              ids[1] * oInfo.strides[1] + ids[0] + oInfo.offset;

    const uint id_dim_out = ids[kDim];

    ids[kDim] = ids[kDim] * get_local_size(1) + lidy;

    iData += ids[3] * iInfo.strides[3] + ids[2] * iInfo.strides[2] +
             ids[1] * iInfo.strides[1] + ids[0] + iInfo.offset;

    if (!IS_FIRST) {
        ilData += ids[3] * iInfo.strides[3] + ids[2] * iInfo.strides[2] +
                  ids[1] * iInfo.strides[1] + ids[0] + iInfo.offset;
    }

    const uint id_dim_in   = ids[kDim];
    const uint istride_dim = iInfo.strides[kDim];

    bool is_valid = (ids[0] < iInfo.dims[0]) && (ids[1] < iInfo.dims[1]) &&
                    (ids[2] < iInfo.dims[2]) && (ids[3] < iInfo.dims[3]);

    local T s_val[THREADS_X * DIMY];
    local uint s_idx[THREADS_X * DIMY];

    T out_val    = init;
    uint out_idx = id_dim_in;

    uint lim = rlenptr ? *rlenptr : iInfo.dims[kDim];
    lim      = (IS_FIRST) ? min((uint)iInfo.dims[kDim], lim) : lim;
    bool within_ragged_bounds =
        (IS_FIRST) ? (out_idx < lim)
                   : ((rlenptr) ? (is_valid) && (*ilData < lim) : true);
    if (is_valid && id_dim_in < iInfo.dims[kDim] && within_ragged_bounds) {
        out_val = *iData;
        if (!IS_FIRST) out_idx = *ilData;
    }

    const uint id_dim_in_start = id_dim_in + group_dim * get_local_size(1);

    for (int id = id_dim_in_start; is_valid && (id < lim);
         id += group_dim * get_local_size(1)) {
        iData = iData + group_dim * get_local_size(1) * istride_dim;

#if IS_FIRST
        binOp(&out_val, &out_idx, *iData, id);
#else
        ilData = ilData + group_dim * get_local_size(1) * istride_dim;
        binOp(&out_val, &out_idx, *iData, *ilData);
#endif
    }

    s_val[lid] = out_val;
    s_idx[lid] = out_idx;

    local T *s_vptr    = s_val + lid;
    local uint *s_iptr = s_idx + lid;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (DIMY == 8) {
        if (lidy < 4) {
            binOp(&out_val, &out_idx, s_vptr[THREADS_X * 4],
                  s_iptr[THREADS_X * 4]);
            *s_vptr = out_val;
            *s_iptr = out_idx;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (DIMY >= 4) {
        if (lidy < 2) {
            binOp(&out_val, &out_idx, s_vptr[THREADS_X * 2],
                  s_iptr[THREADS_X * 2]);
            *s_vptr = out_val;
            *s_iptr = out_idx;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (DIMY >= 2) {
        if (lidy < 1) {
            binOp(&out_val, &out_idx, s_vptr[THREADS_X * 1],
                  s_iptr[THREADS_X * 1]);
            *s_vptr = out_val;
            *s_iptr = out_idx;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lidy == 0 && is_valid && (id_dim_out < oInfo.dims[kDim])) {
        *oData  = *s_vptr;
        *olData = *s_iptr;
    }
}
