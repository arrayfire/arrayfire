#if Ti == double
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

__kernel
void scan_first_kernel(__global To *oData, KParam oInfo,
                       __global To *tData, KParam tInfo,
                       const __global Ti *iData, KParam iInfo,
                       uint groups_x, uint groups_y,
                       uint lim)
{

    const uint lidx = get_local_id(0);
    const uint lidy = get_local_id(1);
    const  int lid  = lidy * get_local_size(0) + lidx;

    const uint zid = get_group_id(0) / groups_x;
    const uint wid = get_group_id(1) / groups_y;
    const uint groupId_x = get_group_id(0) - (groups_x) * zid;
    const uint groupId_y = get_group_id(1) - (groups_y) * wid;
    const uint xid = groupId_x * get_local_size(0) * lim + lidx;
    const uint yid = groupId_y * get_local_size(1) + lidy;

    iData += wid * iInfo.strides[3] + zid * iInfo.strides[2] +
        yid * iInfo.strides[1] + iInfo.offset;

    tData += wid * tInfo.strides[3] + zid * tInfo.strides[2] +
        yid * tInfo.strides[1] + tInfo.offset;

    oData += wid * oInfo.strides[3] + zid * oInfo.strides[2] +
        yid * oInfo.strides[1] + oInfo.offset;

    bool cond_yzw = (yid < oInfo.dims[1]) && (zid < oInfo.dims[2]) && (wid < oInfo.dims[3]);

    __local To l_val0[SHARED_MEM_SIZE];
    __local To l_val1[SHARED_MEM_SIZE];
    __local To *l_val = l_val0;
    __local To l_tmp[DIMY];

    bool flip = 0;

    const To init_val = init;
    uint id = xid;
    To val = init_val;

    const bool isLast = (lidx == (DIMX - 1));

    for (int k = 0; k < lim; k++) {

        if (isLast) l_tmp[lidy] = val;

        bool cond = (cond_yzw && (id < oInfo.dims[0]));
        val = cond ? transform(iData[id]) : init_val;
        l_val[lid] = val;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int off = 1; off < DIMX; off *= 2) {
            if (lidx >= off) val = binOp(val, l_val[lid - off]);

            flip = 1 - flip;
            l_val = flip ? l_val1 : l_val0;
            l_val[lid] = val;
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        val = binOp(val, l_tmp[lidy]);
        if (cond) oData[id] = val;
        id += DIMX;
    }

    if (!isFinalPass && isLast) {
        tData[groupId_x] = val;
    }
}

__kernel
void bcast_first_kernel(__global To *oData, KParam oInfo,
                        const __global To *tData, KParam tInfo,
                        uint groups_x, uint groups_y, uint lim)
{
    const uint lidx = get_local_id(0);
    const uint lidy = get_local_id(1);
    const uint lid  = lidy * get_local_size(0) + lidx;

    const uint zid = get_group_id(0) / groups_x;
    const uint wid = get_group_id(1) / groups_y;
    const uint groupId_x = get_group_id(0) - (groups_x) * zid;
    const uint groupId_y = get_group_id(1) - (groups_y) * wid;
    const uint xid = groupId_x * get_local_size(0) * lim + lidx;
    const uint yid = groupId_y * get_local_size(1) + lidy;

    tData += wid * tInfo.strides[3] + zid * tInfo.strides[2] +
        yid * tInfo.strides[1] + tInfo.offset;

    oData += wid * oInfo.strides[3] + zid * oInfo.strides[2] +
        yid * oInfo.strides[1] + oInfo.offset;

    bool cond = (yid < oInfo.dims[1]) && (zid < oInfo.dims[2]) && (wid < oInfo.dims[3]);

    if (!cond) return;
    if (groupId_x == 0) return;

    To accum = tData[groupId_x - 1];

    for (int k = 0, id = xid;
         k < lim && id < oInfo.dims[0];
         k++, id += DIMX) {

        oData[id] = binOp(accum, oData[id]);
    }

}
