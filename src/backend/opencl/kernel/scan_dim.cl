#if Ti == double
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

__kernel
void scan_dim_kernel(__global To *oData, KParam oInfo,
                     __global To *tData, KParam tInfo,
                     const __global Ti *iData, KParam iInfo,
                     uint groups_x,
                     uint groups_y,
                     uint groups_dim,
                     uint lim)
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
    // There are DIMY elements per group for in
    // Hence increment ids[dim] just after offseting out and before offsetting in
    tData += ids[3] * tInfo.strides[3] + ids[2] * tInfo.strides[2] + ids[1] * tInfo.strides[1] + ids[0];
    const uint groupId_dim = ids[dim];

    ids[dim] = ids[dim] * DIMY * lim + lidy;
    oData  += ids[3] * oInfo.strides[3] + ids[2] * oInfo.strides[2] + ids[1] * oInfo.strides[1] + ids[0];
    iData  += ids[3] *  iInfo.strides[3] + ids[2] *  iInfo.strides[2] + ids[1] *  iInfo.strides[1] + ids[0];
    uint id_dim = ids[dim];
    const uint out_dim = oInfo.dims[dim];

    bool is_valid =
        (ids[0] < oInfo.dims[0]) &&
        (ids[1] < oInfo.dims[1]) &&
        (ids[2] < oInfo.dims[2]) &&
        (ids[3] < oInfo.dims[3]);

    const uint ostride_dim = oInfo.strides[dim];
    const uint istride_dim =  iInfo.strides[dim];

    __local To l_val[THREADS_X * DIMY * 2];
    __local To l_tmp[THREADS_X];

    __local To *l_ptr =  l_val + lid;

    const To init_val  = init;
    To val = init_val;
    const bool isLast = (lidy == (DIMY - 1));

    for (int k = 0; k < lim; k++) {

        if (isLast) l_tmp[lidx] = val;

        bool cond = (is_valid) && (id_dim < out_dim);
        val = cond ? transform(*iData) : init_val;
        *l_ptr = val;
        barrier(CLK_LOCAL_MEM_FENCE);

        uint start = 0;
        for (int off = 1; off < DIMY; off *= 2) {

            if (lidy >= off) val = binOp(val, l_ptr[(start - off) * THREADS_X]);
            start = DIMY - start;
            l_ptr[start * THREADS_X] = val;

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        val = binOp(val, l_tmp[lidx]);
        if (cond) *oData = val;

        id_dim += DIMY;
        iData += DIMY * istride_dim;
        oData += DIMY * ostride_dim;
    }

    if (!isFinalPass &&
        is_valid &&
        (groupId_dim < tInfo.dims[dim]) &&
        isLast) {
        *tData = val;
    }
}

__kernel
void bcast_dim_kernel(__global To *oData, KParam oInfo,
                      const __global To *tData, KParam tInfo,
                      uint groups_x,
                      uint groups_y,
                      uint groups_dim,
                      uint lim)
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
    // There are DIMY elements per group for in
    // Hence increment ids[dim] just after offseting out and before offsetting in
    tData += ids[3] * tInfo.strides[3] + ids[2] * tInfo.strides[2] + ids[1] * tInfo.strides[1] + ids[0];
    const uint groupId_dim = ids[dim];

    ids[dim] = ids[dim] * DIMY * lim + lidy;
    oData  += ids[3] * oInfo.strides[3] + ids[2] * oInfo.strides[2] + ids[1] * oInfo.strides[1] + ids[0];

    const uint id_dim = ids[dim];
    const uint out_dim = oInfo.dims[dim];

    bool is_valid =
        (ids[0] < oInfo.dims[0]) &&
        (ids[1] < oInfo.dims[1]) &&
        (ids[2] < oInfo.dims[2]) &&
        (ids[3] < oInfo.dims[3]);

    if (!is_valid) return;
    if (groupId_dim == 0) return;

    To accum = *(tData - tInfo.strides[dim]);

    const uint ostride_dim = oInfo.strides[dim];

    for (int k = 0, id = id_dim;
         is_valid && k < lim && (id < out_dim);
         k++, id += DIMY) {

        *oData = binOp(*oData, accum);
        oData += DIMY * ostride_dim;
    }
}
