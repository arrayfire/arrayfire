/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

char calculate_head_flags_dim(const __global Tk *kptr, int id, int stride) {
    return (id == 0) ? 1 : ((*kptr) != (*(kptr - stride)));
}

__kernel void scan_dim_by_key_nonfinal_kernel(
    __global To *oData, KParam oInfo, __global To *tData, KParam tInfo,
    __global char *tfData, KParam tfInfo, __global int *tiData, KParam tiInfo,
    const __global Ti *iData, KParam iInfo, const __global Tk *kData,
    KParam kInfo, uint groups_x, uint groups_y, uint groups_dim, uint lim) {
    const int lidx = get_local_id(0);
    const int lidy = get_local_id(1);
    const int lid  = lidy * THREADS_X + lidx;

    const int zid       = get_group_id(0) / groups_x;
    const int wid       = get_group_id(1) / groups_y;
    const int groupId_x = get_group_id(0) - (groups_x)*zid;
    const int groupId_y = get_group_id(1) - (groups_y)*wid;
    const int xid       = groupId_x * get_local_size(0) + lidx;
    const int yid       = groupId_y;

    int ids[4] = {xid, yid, zid, wid};

    // There is only one element per group for out
    // There are DIMY elements per group for in
    // Hence increment ids[dim] just after offseting out and before offsetting
    // in
    tData += ids[3] * tInfo.strides[3] + ids[2] * tInfo.strides[2] +
             ids[1] * tInfo.strides[1] + ids[0];
    tfData += ids[3] * tfInfo.strides[3] + ids[2] * tfInfo.strides[2] +
              ids[1] * tfInfo.strides[1] + ids[0];
    tiData += ids[3] * tiInfo.strides[3] + ids[2] * tiInfo.strides[2] +
              ids[1] * tiInfo.strides[1] + ids[0];
    const int groupId_dim = ids[dim];

    ids[dim] = ids[dim] * DIMY * lim + lidy;
    oData += ids[3] * oInfo.strides[3] + ids[2] * oInfo.strides[2] +
             ids[1] * oInfo.strides[1] + ids[0];
    iData += ids[3] * iInfo.strides[3] + ids[2] * iInfo.strides[2] +
             ids[1] * iInfo.strides[1] + ids[0];
    kData += ids[3] * kInfo.strides[3] + ids[2] * kInfo.strides[2] +
             ids[1] * kInfo.strides[1] + ids[0];
    iData += iInfo.offset;

    int id_dim        = ids[dim];
    const int out_dim = oInfo.dims[dim];

    bool is_valid = (ids[0] < oInfo.dims[0]) && (ids[1] < oInfo.dims[1]) &&
                    (ids[2] < oInfo.dims[2]) && (ids[3] < oInfo.dims[3]);

    const int ostride_dim = oInfo.strides[dim];
    const int istride_dim = iInfo.strides[dim];

    __local To l_val0[THREADS_X * DIMY];
    __local To l_val1[THREADS_X * DIMY];
    __local char l_flg0[THREADS_X * DIMY];
    __local char l_flg1[THREADS_X * DIMY];
    __local To *l_val   = l_val0;
    __local char *l_flg = l_flg0;
    __local To l_tmp[THREADS_X];
    __local char l_ftmp[THREADS_X];
    __local int boundaryid[THREADS_X];

    bool flip         = 0;
    const To init_val = init;
    To val            = init_val;
    const bool isLast = (lidy == (DIMY - 1));

    if (isLast) {
        l_tmp[lidx]      = val;
        l_ftmp[lidx]     = 0;
        boundaryid[lidx] = -1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    char flag = 0;
    for (int k = 0; k < lim; k++) {
        bool cond = (is_valid) && (id_dim < out_dim);

        if (cond) {
            flag = calculate_head_flags_dim(kData, id_dim, kInfo.strides[dim]);
        } else {
            flag = 0;
        }

        // Load val from global in
        if (inclusive_scan) {
            if (!cond) {
                val = init_val;
            } else {
                val = transform(*iData);
            }
        } else {
            if ((id_dim == 0) || (!cond) || flag) {
                val = init_val;
            } else {
                val = transform(*(iData - iInfo.strides[dim]));
            }
        }

        // Add partial result from last iteration before scan operation
        if ((lidy == 0) && (flag == 0)) {
            val  = binOp(val, l_tmp[lidx]);
            flag = l_ftmp[lidx];
        }

        // Write to shared memory
        l_val[lid] = val;
        l_flg[lid] = flag;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Segmented Scan
        for (int off = 1; off < DIMY; off *= 2) {
            if (lidy >= off) {
                val =
                    l_flg[lid] ? val : binOp(val, l_val[lid - off * THREADS_X]);
                flag = l_flg[lid] | l_flg[lid - off * THREADS_X];
            }
            flip       = 1 - flip;
            l_val      = flip ? l_val1 : l_val0;
            l_flg      = flip ? l_flg1 : l_flg0;
            l_val[lid] = val;
            l_flg[lid] = flag;
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Identify segment boundary
        if (lidy == 0) {
            if ((l_ftmp[lidx] == 0) && (l_flg[lid] == 1)) {
                boundaryid[lidx] = id_dim;
            }
        } else {
            if ((l_flg[lid - THREADS_X] == 0) && (l_flg[lid] == 1)) {
                boundaryid[lidx] = id_dim;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (cond) *oData = val;
        if (isLast) {
            l_tmp[lidx]  = val;
            l_ftmp[lidx] = flag;
        }
        id_dim += DIMY;
        kData += DIMY * kInfo.strides[dim];
        iData += DIMY * istride_dim;
        oData += DIMY * ostride_dim;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (is_valid && (groupId_dim < tInfo.dims[dim]) && isLast) {
        *tData       = val;
        *tfData      = flag;
        int boundary = boundaryid[lidx];
        *tiData      = (boundary == -1) ? id_dim : boundary;
    }
}

__kernel void scan_dim_by_key_final_kernel(
    __global To *oData, KParam oInfo, const __global Ti *iData, KParam iInfo,
    const __global Tk *kData, KParam kInfo, uint groups_x, uint groups_y,
    uint groups_dim, uint lim) {
    const int lidx = get_local_id(0);
    const int lidy = get_local_id(1);
    const int lid  = lidy * THREADS_X + lidx;

    const int zid       = get_group_id(0) / groups_x;
    const int wid       = get_group_id(1) / groups_y;
    const int groupId_x = get_group_id(0) - (groups_x)*zid;
    const int groupId_y = get_group_id(1) - (groups_y)*wid;
    const int xid       = groupId_x * get_local_size(0) + lidx;
    const int yid       = groupId_y;

    int ids[4] = {xid, yid, zid, wid};

    // There is only one element per group for out
    // There are DIMY elements per group for in
    // Hence increment ids[dim] just after offseting out and before offsetting
    // in
    const int groupId_dim = ids[dim];

    ids[dim] = ids[dim] * DIMY * lim + lidy;
    oData += ids[3] * oInfo.strides[3] + ids[2] * oInfo.strides[2] +
             ids[1] * oInfo.strides[1] + ids[0];
    iData += ids[3] * iInfo.strides[3] + ids[2] * iInfo.strides[2] +
             ids[1] * iInfo.strides[1] + ids[0];
    kData += ids[3] * kInfo.strides[3] + ids[2] * kInfo.strides[2] +
             ids[1] * kInfo.strides[1] + ids[0];
    iData += iInfo.offset;

    int id_dim        = ids[dim];
    const int out_dim = oInfo.dims[dim];

    bool is_valid = (ids[0] < oInfo.dims[0]) && (ids[1] < oInfo.dims[1]) &&
                    (ids[2] < oInfo.dims[2]) && (ids[3] < oInfo.dims[3]);

    const int ostride_dim = oInfo.strides[dim];
    const int istride_dim = iInfo.strides[dim];

    __local To l_val0[THREADS_X * DIMY];
    __local To l_val1[THREADS_X * DIMY];
    __local char l_flg0[THREADS_X * DIMY];
    __local char l_flg1[THREADS_X * DIMY];
    __local To *l_val   = l_val0;
    __local char *l_flg = l_flg0;
    __local To l_tmp[THREADS_X];
    __local char l_ftmp[THREADS_X];

    bool flip         = 0;
    const To init_val = init;
    To val            = init_val;
    const bool isLast = (lidy == (DIMY - 1));

    if (isLast) {
        l_tmp[lidx]  = val;
        l_ftmp[lidx] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    char flag = 0;
    for (int k = 0; k < lim; k++) {
        bool cond = (is_valid) && (id_dim < out_dim);

        if (calculateFlags) {
            if (cond) {
                flag =
                    calculate_head_flags_dim(kData, id_dim, kInfo.strides[dim]);
            } else {
                flag = 0;
            }
        } else {
            flag = *kData;
        }

        // Load val from global in
        if (inclusive_scan) {
            if (!cond) {
                val = init_val;
            } else {
                val = transform(*iData);
            }
        } else {
            if ((id_dim == 0) || (!cond) || flag) {
                val = init_val;
            } else {
                val = transform(*(iData - iInfo.strides[dim]));
            }
        }

        // Add partial result from last iteration before scan operation
        if ((lidy == 0) && (flag == 0)) {
            val  = binOp(val, l_tmp[lidx]);
            flag = l_ftmp[lidx];
        }

        // Write to shared memory
        l_val[lid] = val;
        l_flg[lid] = flag;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Segmented Scan
        for (int off = 1; off < DIMY; off *= 2) {
            if (lidy >= off) {
                val =
                    l_flg[lid] ? val : binOp(val, l_val[lid - off * THREADS_X]);
                flag = l_flg[lid] | l_flg[lid - off * THREADS_X];
            }
            flip       = 1 - flip;
            l_val      = flip ? l_val1 : l_val0;
            l_flg      = flip ? l_flg1 : l_flg0;
            l_val[lid] = val;
            l_flg[lid] = flag;
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (cond) *oData = val;
        if (isLast) {
            l_tmp[lidx]  = val;
            l_ftmp[lidx] = flag;
        }
        id_dim += DIMY;
        kData += DIMY * kInfo.strides[dim];
        iData += DIMY * istride_dim;
        oData += DIMY * ostride_dim;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void bcast_dim_kernel(__global To *oData, KParam oInfo,
                               const __global To *tData, KParam tInfo,
                               const __global int *tiData, KParam tiInfo,
                               uint groups_x, uint groups_y, uint groups_dim,
                               uint lim) {
    const int lidx = get_local_id(0);
    const int lidy = get_local_id(1);
    const int lid  = lidy * THREADS_X + lidx;

    const int zid       = get_group_id(0) / groups_x;
    const int wid       = get_group_id(1) / groups_y;
    const int groupId_x = get_group_id(0) - (groups_x)*zid;
    const int groupId_y = get_group_id(1) - (groups_y)*wid;
    const int xid       = groupId_x * get_local_size(0) + lidx;
    const int yid       = groupId_y;

    int ids[4]            = {xid, yid, zid, wid};
    const int groupId_dim = ids[dim];

    if (groupId_dim != 0) {
        // There is only one element per group for out
        // There are DIMY elements per group for in
        // Hence increment ids[dim] just after offseting out and before
        // offsetting in
        tiData += ids[3] * tiInfo.strides[3] + ids[2] * tiInfo.strides[2] +
                  ids[1] * tiInfo.strides[1] + ids[0];
        tData += ids[3] * tInfo.strides[3] + ids[2] * tInfo.strides[2] +
                 ids[1] * tInfo.strides[1] + ids[0];

        ids[dim] = ids[dim] * DIMY * lim + lidy;
        oData += ids[3] * oInfo.strides[3] + ids[2] * oInfo.strides[2] +
                 ids[1] * oInfo.strides[1] + ids[0];

        const int id_dim = ids[dim];

        bool is_valid = (ids[0] < oInfo.dims[0]) && (ids[1] < oInfo.dims[1]) &&
                        (ids[2] < oInfo.dims[2]) && (ids[3] < oInfo.dims[3]);

        if (is_valid) {
            int boundary = *tiData;
            To accum     = *(tData - tInfo.strides[dim]);

            const int ostride_dim = oInfo.strides[dim];

            for (int k = 0, id = id_dim; is_valid && k < lim && (id < boundary);
                 k++, id += DIMY) {
                *oData = binOp(*oData, accum);
                oData += DIMY * ostride_dim;
            }
        }
    }
}
