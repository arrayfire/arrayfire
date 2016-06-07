/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

static char calculate_head_flags(const __global Tk *kptr, int id, int previd)
{
    char flag;
    if (id == 0) {
        flag = 1;
    } else {
        flag = (kptr[id] != kptr[previd]);
    }
    return flag;
}

void scan_first_by_key_core(const bool invalid,
        To *val, char *flag,
        const __global Ti *in, const To init_val,
        __local To *l_val0, __local To *l_val1,
        __local char *l_flg0, __local char *l_flg1,
        __local To *last_val, __local char *last_flag,
        const int lid, const int lidx, const int lidy)
{
    bool flip = 0;
    __local To *l_val = l_val0;
    __local char *l_flg = l_flg0;
    *val = invalid? init_val : transform(*in);

    if ((lidx == 0) && (flag == 0)) {
        *val = binOp(*val, last_val[lidy]);
        *flag = *flag | last_flag[lidy];
    }

    l_val0[lid] = *val;
    l_flg0[lid] = *flag;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int off = 1; off < DIMX; off *= 2) {

        if (lidx >= off) {
            *val = l_flg[lid] ? *val : binOp(*val, l_val[lid - off]);
            *flag = l_flg[lid] | l_flg[lid - off];
        }
        flip = 1 - flip;
        l_val = flip ? l_val1 : l_val0;
        l_flg = flip ? l_flg1 : l_flg0;
        l_val[lid] = *val;
        l_flg[lid] = *flag;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel
void scan_first_by_key_nonfinal_kernel(__global To *oData, KParam oInfo,
                       __global To *tData, KParam tInfo,
                       __global char *tfData, KParam tfInfo,
                       __global int *tiData, KParam tiInfo,
                       const __global Ti *iData, KParam iInfo,
                       const __global Tk *kData, KParam kInfo,
                       uint groups_x, uint groups_y,
                       uint lim)
{
    const int lidx = get_local_id(0);
    const int lidy = get_local_id(1);
    const int lid  = lidy * get_local_size(0) + lidx;

    const int zid = get_group_id(0) / groups_x;
    const int wid = get_group_id(1) / groups_y;
    const int groupId_x = get_group_id(0) - (groups_x) * zid;
    const int groupId_y = get_group_id(1) - (groups_y) * wid;
    const int xid = groupId_x * get_local_size(0) * lim + lidx;
    const int yid = groupId_y * get_local_size(1) + lidy;

    bool cond_yzw = (yid < oInfo.dims[1]) && (zid < oInfo.dims[2]) && (wid < oInfo.dims[3]);

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

    __local To l_val0[SHARED_MEM_SIZE];
    __local To l_val1[SHARED_MEM_SIZE];
    __local char l_flg0[SHARED_MEM_SIZE];
    __local char l_flg1[SHARED_MEM_SIZE];
    __local To last_val[DIMY];
    __local char last_flag[DIMY];
    __local int boundaryid;

    const To init_val = init;
    int id = xid;
    To val = init_val;
    const bool isLast = (lidx == (DIMX - 1));

    char flag = 0;
    if (!inclusive_scan) {
        iData -= 1;
    }

    if (isLast) {
        last_val[lidy] = val;
        last_flag[lidy] = 0;
        boundaryid = -1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    __local char *prev;
    if (lidx == 0) {
        prev = &last_flag[lidy];
    } else {
        prev = &l_flg0[lidx-1];
    }
    __local char *curr = &l_flg0[lidx];

    for (int k = 0; k < lim; k++) {

        bool cond = ((id < oInfo.dims[0]) && cond_yzw);

        if (cond) {
            flag = calculate_head_flags(kData, id, id - 1);
        } else {
            flag = 0;
        }

        bool invalid = !cond;
        if (!inclusive_scan) invalid = invalid || (id == 0) || flag;

        scan_first_by_key_core(invalid, &val, &flag, iData, init_val,
                l_val0, l_val1, l_flg0, l_flg1, last_val, last_flag,
                lid, lidx, lidy);

        if ((*prev == 0) && (*curr == 1)) {
            boundaryid = id;
        }

        if (cond) oData[id] = val;
        if (isLast) {
            last_val[lidy] = val;
            last_flag[lidy] = flag;
        }
        id += DIMX;
        barrier(CLK_LOCAL_MEM_FENCE); //FIXME: May be needed only for non nvidia gpus
    }

    if (isLast && cond_yzw) {
        tData[groupId_x] = val;
        tfData[groupId_x] = flag;
        tiData[groupId_x] = boundaryid;
    }
}

__kernel
void scan_first_by_key_final_kernel(__global To *oData, KParam oInfo,
                       const __global Ti *iData, KParam iInfo,
                       const __global Tk *kData, KParam kInfo,
                       uint groups_x, uint groups_y,
                       uint lim)
{
    const int lidx = get_local_id(0);
    const int lidy = get_local_id(1);
    const int lid  = lidy * get_local_size(0) + lidx;

    const int zid = get_group_id(0) / groups_x;
    const int wid = get_group_id(1) / groups_y;
    const int groupId_x = get_group_id(0) - (groups_x) * zid;
    const int groupId_y = get_group_id(1) - (groups_y) * wid;
    const int xid = groupId_x * get_local_size(0) * lim + lidx;
    const int yid = groupId_y * get_local_size(1) + lidy;

    bool cond_yzw = (yid < oInfo.dims[1]) && (zid < oInfo.dims[2]) && (wid < oInfo.dims[3]);

    iData += wid * iInfo.strides[3] + zid * iInfo.strides[2] +
        yid * iInfo.strides[1] + iInfo.offset;

    kData += wid * kInfo.strides[3] + zid * kInfo.strides[2] +
        yid * kInfo.strides[1] + kInfo.offset;

    oData += wid * oInfo.strides[3] + zid * oInfo.strides[2] +
        yid * oInfo.strides[1] + oInfo.offset;

    __local To l_val0[SHARED_MEM_SIZE];
    __local To l_val1[SHARED_MEM_SIZE];
    __local char l_flg0[SHARED_MEM_SIZE];
    __local char l_flg1[SHARED_MEM_SIZE];
    __local To last_val[DIMY];
    __local char last_flag[DIMY];

    const To init_val = init;
    int id = xid;
    To val = init_val;
    const bool isLast = (lidx == (DIMX - 1));

    if (!inclusive_scan) {
        iData -= 1;
    }

    if (isLast) {
        last_val[lidy] = val;
        last_flag[lidy] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

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

        bool invalid = !cond;
        if (!inclusive_scan) invalid = invalid || (id == 0) || flag;

        scan_first_by_key_core(invalid, &val, &flag, iData, init_val,
                l_val0, l_val1, l_flg0, l_flg1, last_val, last_flag,
                lid, lidx, lidy);

        if (cond) oData[id] = val;
        if (isLast) {
            last_val[lidy] = val;
            last_flag[lidy] = flag;
        }
        id += DIMX;
        barrier(CLK_LOCAL_MEM_FENCE); //FIXME: May be needed only for non nvidia gpus
    }
}

__kernel
void bcast_first_kernel(__global To *oData, KParam oInfo,
                        const __global To *tData, KParam tInfo,
                        const __global int *tiData, KParam tiInfo,
                        uint groups_x, uint groups_y, uint lim)
{
    const int lidx = get_local_id(0);
    const int lidy = get_local_id(1);
    const int lid  = lidy * get_local_size(0) + lidx;

    const int zid = get_group_id(0) / groups_x;
    const int wid = get_group_id(1) / groups_y;
    const int groupId_x = get_group_id(0) - (groups_x) * zid;
    const int groupId_y = get_group_id(1) - (groups_y) * wid;
    const int xid = groupId_x * get_local_size(0) * lim + lidx;
    const int yid = groupId_y * get_local_size(1) + lidy;

    if (groupId_x != 0) {
        bool cond = (yid < oInfo.dims[1]) && (zid < oInfo.dims[2]) && (wid < oInfo.dims[3]);

        if (cond) {

            tiData += wid * tiInfo.strides[3] + zid * tiInfo.strides[2] +
                yid * tiInfo.strides[1] + tiInfo.offset;

            tData += wid * tInfo.strides[3] + zid * tInfo.strides[2] +
                yid * tInfo.strides[1] + tInfo.offset;

            oData += wid * oInfo.strides[3] + zid * oInfo.strides[2] +
                yid * oInfo.strides[1] + oInfo.offset;

            int boundary = tiData[groupId_x];
            To accum = tData[groupId_x - 1];

            for (int k = 0, id = xid;
                 k < lim && id < boundary;
                 k++, id += DIMX) {

                oData[id] = binOp(accum, oData[id]);
            }
        }
    }
}
