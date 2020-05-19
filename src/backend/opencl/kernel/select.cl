/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#ifndef flip
#define flip 0
#endif

#ifndef is_same
#define is_same 0
#endif

int getOffset(dim_t *dims, dim_t *strides, dim_t *refdims, int ids[4]) {
    int off = 0;
    off += ids[3] * (dims[3] == refdims[3]) * strides[3];
    off += ids[2] * (dims[2] == refdims[2]) * strides[2];
    off += ids[1] * (dims[1] == refdims[1]) * strides[1];
    return off;
}

kernel void select_kernel(global T *optr, KParam oinfo,
                            global char *cptr_, KParam cinfo,
                            global T *aptr_, KParam ainfo, __global T *bptr_,
                            KParam binfo, int groups_0, int groups_1) {
    global char *cptr = cptr_ + cinfo.offset;
    global T *aptr    = aptr_ + ainfo.offset;
    global T *bptr    = bptr_ + binfo.offset;

    const int idz = get_group_id(0) / groups_0;
    const int idw = get_group_id(1) / groups_1;

    const int group_id_0 = get_group_id(0) - idz * groups_0;
    const int group_id_1 = get_group_id(1) - idw * groups_1;

    const int idx0 = group_id_0 * get_local_size(0) + get_local_id(0);
    const int idy  = group_id_1 * get_local_size(1) + get_local_id(1);

    const int off = idw * oinfo.strides[3] + idz * oinfo.strides[2] +
                    idy * oinfo.strides[1];

    if (idw >= oinfo.dims[3] || idz >= oinfo.dims[2] || idy >= oinfo.dims[1]) {
        return;
    }

    int ids[] = {idx0, idy, idz, idw};

    optr += off;
    aptr += getOffset(ainfo.dims, ainfo.strides, oinfo.dims, ids);
    bptr += getOffset(binfo.dims, binfo.strides, oinfo.dims, ids);
    cptr += getOffset(cinfo.dims, cinfo.strides, oinfo.dims, ids);

    if (is_same) {
        for (int idx = idx0; idx < oinfo.dims[0];
             idx += get_local_size(0) * groups_0) {
            optr[idx] = (cptr[idx]) ? aptr[idx] : bptr[idx];
        }
    } else {
        bool csame = cinfo.dims[0] == oinfo.dims[0];
        bool asame = ainfo.dims[0] == oinfo.dims[0];
        bool bsame = binfo.dims[0] == oinfo.dims[0];
        for (int idx = idx0; idx < oinfo.dims[0];
             idx += get_local_size(0) * groups_0) {
            optr[idx] =
                (cptr[csame * idx]) ? aptr[asame * idx] : bptr[bsame * idx];
        }
    }
}

kernel void select_scalar_kernel(global T *optr, KParam oinfo,
                                   global char *cptr_, KParam cinfo,
                                   global T *aptr_, KParam ainfo, T b,
                                   int groups_0, int groups_1) {
    global char *cptr = cptr_ + cinfo.offset;
    global T *aptr    = aptr_ + ainfo.offset;

    const int idz = get_group_id(0) / groups_0;
    const int idw = get_group_id(1) / groups_1;

    const int group_id_0 = get_group_id(0) - idz * groups_0;
    const int group_id_1 = get_group_id(1) - idw * groups_1;

    const int idx0 = group_id_0 * get_local_size(0) + get_local_id(0);
    const int idy  = group_id_1 * get_local_size(1) + get_local_id(1);

    const int off = idw * oinfo.strides[3] + idz * oinfo.strides[2] +
                    idy * oinfo.strides[1];

    int ids[] = {idx0, idy, idz, idw};
    optr += off;
    aptr += getOffset(ainfo.dims, ainfo.strides, oinfo.dims, ids);
    cptr += getOffset(cinfo.dims, cinfo.strides, oinfo.dims, ids);

    if (idw >= oinfo.dims[3] || idz >= oinfo.dims[2] || idy >= oinfo.dims[1]) {
        return;
    }

    for (int idx = idx0; idx < oinfo.dims[0];
         idx += get_local_size(0) * groups_0) {
        optr[idx] = (cptr[idx] ^ flip) ? aptr[idx] : b;
    }
}
