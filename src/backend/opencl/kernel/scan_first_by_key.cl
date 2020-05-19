/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

char calculate_head_flags(const global Tk *kptr, int id, int previd) {
    return (id == 0) ? 1 : (kptr[id] != kptr[previd]);
}

kernel void scanFirstByKeyNonfinal(global To *oData, KParam oInfo,
                                   global To *tData, KParam tInfo,
                                   global char *tfData, KParam tfInfo,
                                   global int *tiData, KParam tiInfo,
                                   const global Ti *iData, KParam iInfo,
                                   const global Tk *kData, KParam kInfo,
                                   uint groups_x, uint groups_y, uint lim) {
    const int lidx = get_local_id(0);
    const int lidy = get_local_id(1);
    const int lid  = lidy * get_local_size(0) + lidx;

    const int zid       = get_group_id(0) / groups_x;
    const int wid       = get_group_id(1) / groups_y;
    const int groupId_x = get_group_id(0) - (groups_x)*zid;
    const int groupId_y = get_group_id(1) - (groups_y)*wid;
    const int xid       = groupId_x * get_local_size(0) * lim + lidx;
    const int yid       = groupId_y * get_local_size(1) + lidy;

    bool cond_yzw =
        (yid < oInfo.dims[1]) && (zid < oInfo.dims[2]) && (wid < oInfo.dims[3]);

    iData += wid * iInfo.strides[3] + zid * iInfo.strides[2] +
             yid * iInfo.strides[1] + iInfo.offset;

    kData += wid * kInfo.strides[3] + zid * kInfo.strides[2] +
             yid * kInfo.strides[1] + kInfo.offset;

    tData += wid * tInfo.strides[3] + zid * tInfo.strides[2] +
             yid * tInfo.strides[1] + tInfo.offset;

    tfData += wid * tfInfo.strides[3] + zid * tfInfo.strides[2] +
              yid * tfInfo.strides[1] + tfInfo.offset;

    tiData += wid * tiInfo.strides[3] + zid * tiInfo.strides[2] +
              yid * tiInfo.strides[1] + tiInfo.offset;

    oData += wid * oInfo.strides[3] + zid * oInfo.strides[2] +
             yid * oInfo.strides[1] + oInfo.offset;

    local To l_val0[SHARED_MEM_SIZE];
    local To l_val1[SHARED_MEM_SIZE];
    local char l_flg0[SHARED_MEM_SIZE];
    local char l_flg1[SHARED_MEM_SIZE];
    local To *l_val   = l_val0;
    local char *l_flg = l_flg0;
    local To l_tmp[DIMY];
    local char l_ftmp[DIMY];
    local int boundaryid[DIMY];

    bool flip = 0;

    const To init_val = init;
    int id            = xid;
    To val            = init_val;

    const bool isLast = (lidx == (DIMX - 1));

    if (isLast) {
        l_tmp[lidy]      = val;
        l_ftmp[lidy]     = 0;
        boundaryid[lidy] = -1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    char flag = 0;
    for (int k = 0; k < lim; k++) {
        bool cond = ((id < oInfo.dims[0]) && cond_yzw);

        if (cond) {
            flag = calculate_head_flags(kData, id, id - 1);
        } else {
            flag = 0;
        }

        // Load val from global in
        if (INCLUSIVE_SCAN) {
            if (!cond) {
                val = init_val;
            } else {
                val = transform(iData[id]);
            }
        } else {
            if ((id == 0) || (!cond) || flag) {
                val = init_val;
            } else {
                val = transform(iData[id - 1]);
            }
        }

        // Add partial result from last iteration before scan operation
        if ((lidx == 0) && (flag == 0)) {
            val  = binOp(val, l_tmp[lidy]);
            flag = l_ftmp[lidy];
        }

        // Write to shared memory
        l_val[lid] = val;
        l_flg[lid] = flag;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Segmented Scan
        for (int off = 1; off < DIMX; off *= 2) {
            if (lidx >= off) {
                val  = l_flg[lid] ? val : binOp(val, l_val[lid - off]);
                flag = l_flg[lid] | l_flg[lid - off];
            }
            flip       = 1 - flip;
            l_val      = flip ? l_val1 : l_val0;
            l_flg      = flip ? l_flg1 : l_flg0;
            l_val[lid] = val;
            l_flg[lid] = flag;
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Identify segment boundary
        if (lidx == 0) {
            if ((l_ftmp[lidy] == 0) && (l_flg[lid] == 1)) {
                boundaryid[lidy] = id;
            }
        } else {
            if ((l_flg[lid - 1] == 0) && (l_flg[lid] == 1)) {
                boundaryid[lidy] = id;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (cond) oData[id] = val;
        if (isLast) {
            l_tmp[lidy]  = val;
            l_ftmp[lidy] = flag;
        }
        id += DIMX;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (isLast && cond_yzw) {
        tData[groupId_x]  = val;
        tfData[groupId_x] = flag;
        int boundary      = boundaryid[lidy];
        tiData[groupId_x] = (boundary == -1) ? id : boundary;
    }
}

kernel void scanFirstByKeyFinal(global To *oData, KParam oInfo,
                                const global Ti *iData, KParam iInfo,
                                const global Tk *kData, KParam kInfo,
                                uint groups_x, uint groups_y, uint lim) {
    const int lidx = get_local_id(0);
    const int lidy = get_local_id(1);
    const int lid  = lidy * get_local_size(0) + lidx;

    const int zid       = get_group_id(0) / groups_x;
    const int wid       = get_group_id(1) / groups_y;
    const int groupId_x = get_group_id(0) - (groups_x)*zid;
    const int groupId_y = get_group_id(1) - (groups_y)*wid;
    const int xid       = groupId_x * get_local_size(0) * lim + lidx;
    const int yid       = groupId_y * get_local_size(1) + lidy;

    bool cond_yzw =
        (yid < oInfo.dims[1]) && (zid < oInfo.dims[2]) && (wid < oInfo.dims[3]);

    iData += wid * iInfo.strides[3] + zid * iInfo.strides[2] +
             yid * iInfo.strides[1] + iInfo.offset;

    kData += wid * kInfo.strides[3] + zid * kInfo.strides[2] +
             yid * kInfo.strides[1] + kInfo.offset;

    oData += wid * oInfo.strides[3] + zid * oInfo.strides[2] +
             yid * oInfo.strides[1] + oInfo.offset;

    local To l_val0[SHARED_MEM_SIZE];
    local To l_val1[SHARED_MEM_SIZE];
    local char l_flg0[SHARED_MEM_SIZE];
    local char l_flg1[SHARED_MEM_SIZE];
    local To *l_val   = l_val0;
    local char *l_flg = l_flg0;
    local To l_tmp[DIMY];
    local char l_ftmp[DIMY];

    bool flip = 0;

    const To init_val = init;
    int id            = xid;
    To val            = init_val;

    const bool isLast = (lidx == (DIMX - 1));

    for (int k = 0; k < lim; k++) {
        char flag = 0;

        bool cond = ((id < oInfo.dims[0]) && cond_yzw);

        if (calculateFlags) {
            if (cond) {
                flag = calculate_head_flags(kData, id, id - 1);
            } else {
                flag = 0;
            }
        } else {
            flag = kData[id];
        }

        // Load val from global in
        if (INCLUSIVE_SCAN) {
            if (!cond) {
                val = init_val;
            } else {
                val = transform(iData[id]);
            }
        } else {
            if ((id == 0) || (!cond) || flag) {
                val = init_val;
            } else {
                val = transform(iData[id - 1]);
            }
        }

        // Add partial result from last iteration before scan operation
        if ((lidx == 0) && (flag == 0)) {
            val  = binOp(val, l_tmp[lidy]);
            flag = flag | l_ftmp[lidy];
        }

        // Write to shared memory
        l_val[lid] = val;
        l_flg[lid] = flag;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Write to shared memory
        for (int off = 1; off < DIMX; off *= 2) {
            if (lidx >= off) {
                val  = l_flg[lid] ? val : binOp(val, l_val[lid - off]);
                flag = l_flg[lid] | l_flg[lid - off];
            }
            flip       = 1 - flip;
            l_val      = flip ? l_val1 : l_val0;
            l_flg      = flip ? l_flg1 : l_flg0;
            l_val[lid] = val;
            l_flg[lid] = flag;
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (cond) oData[id] = val;
        if (isLast) {
            l_tmp[lidy]  = val;
            l_ftmp[lidy] = flag;
        }
        id += DIMX;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

kernel void bcastFirstByKey(global To *oData, KParam oInfo,
                            const global To *tData, KParam tInfo,
                            const global int *tiData, KParam tiInfo,
                            uint groups_x, uint groups_y, uint lim) {
    const int lidx = get_local_id(0);
    const int lidy = get_local_id(1);

    const int zid       = get_group_id(0) / groups_x;
    const int wid       = get_group_id(1) / groups_y;
    const int groupId_x = get_group_id(0) - (groups_x)*zid;
    const int groupId_y = get_group_id(1) - (groups_y)*wid;
    const int xid       = groupId_x * get_local_size(0) * lim + lidx;
    const int yid       = groupId_y * get_local_size(1) + lidy;

    if (groupId_x != 0) {
        bool cond = (yid < oInfo.dims[1]) && (zid < oInfo.dims[2]) &&
                    (wid < oInfo.dims[3]);

        if (cond) {
            tiData += wid * tiInfo.strides[3] + zid * tiInfo.strides[2] +
                      yid * tiInfo.strides[1] + tiInfo.offset;

            tData += wid * tInfo.strides[3] + zid * tInfo.strides[2] +
                     yid * tInfo.strides[1] + tInfo.offset;

            oData += wid * oInfo.strides[3] + zid * oInfo.strides[2] +
                     yid * oInfo.strides[1] + oInfo.offset;

            int boundary = tiData[groupId_x];
            To accum     = tData[groupId_x - 1];

            for (int k = 0, id = xid;
                 k < lim && id < oInfo.dims[0] && id < boundary;
                 k++, id += DIMX) {
                oData[id] = binOp(accum, oData[id]);
            }
        }
    }
}
