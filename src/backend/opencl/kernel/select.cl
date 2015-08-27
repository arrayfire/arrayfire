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

int getOffset(dim_t *dims, dim_t *strides, dim_t *refdims)
{
    int off = 0;
    off += (dims[3] == refdims[3]) * strides[3];
    off += (dims[2] == refdims[2]) * strides[2];
    off += (dims[1] == refdims[1]) * strides[1];
    off += (dims[0] == refdims[0]);
    return off;
}

__kernel
void select_kernel(__global T *optr, KParam oinfo,
                   __global char *cptr, KParam cinfo,
                   __global T *aptr, KParam ainfo,
                   __global T *bptr, KParam binfo,
                   int groups_0,
                   int groups_1)
{
    const int idz = get_group_id(0) / groups_0;
    const int idw = get_group_id(1) / groups_1;

    const int group_id_0 = get_group_id(0) - idz * groups_0;
    const int group_id_1 = get_group_id(1) - idz * groups_1;

    const int idx = group_id_0 * get_local_size(0) + get_local_id(0);
    const int idy = group_id_1 * get_local_size(1) + get_local_id(1);

    const int off = idw * oinfo.strides[3] + idz * oinfo.strides[2] + idy * oinfo.strides[1] + idx;

    optr += off;

    if (is_same) {
        aptr += off;
        bptr += off;
        cptr += off;
    } else {
        aptr += getOffset(ainfo.dims, ainfo.strides, oinfo.dims);
        bptr += getOffset(binfo.dims, binfo.strides, oinfo.dims);
        cptr += getOffset(cinfo.dims, cinfo.strides, oinfo.dims);
    }

    if (idx < oinfo.dims[0] && idy < oinfo.dims[1] && idz < oinfo.dims[2] && idw < oinfo.dims[3]) {
        *optr = (*cptr) ? *aptr : *bptr;
    }
}

__kernel
void select_scalar_kernel(__global T *optr, KParam oinfo,
                          __global char *cptr, KParam cinfo,
                          __global T *aptr, KParam ainfo,
                          T b,
                          int groups_0,
                          int groups_1)
{
    const int idz = get_group_id(0) / groups_0;
    const int idw = get_group_id(1) / groups_1;

    const int group_id_0 = get_group_id(0) - idz * groups_0;
    const int group_id_1 = get_group_id(1) - idz * groups_1;

    const int idx = group_id_0 * get_local_size(0) + get_local_id(0);
    const int idy = group_id_1 * get_local_size(1) + get_local_id(1);

    const int off = idw * oinfo.strides[3] + idz * oinfo.strides[2] + idy * oinfo.strides[1] + idx;

    optr += off;
    aptr += off;
    cptr += off;

    if (idx < oinfo.dims[0] && idy < oinfo.dims[1] && idz < oinfo.dims[2] && idw < oinfo.dims[3]) {
        *optr = ((*cptr) ^ flip) ? *aptr : b;
    }
}
