/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

typedef struct {
    dim_t dim[4];
} dims_t;

__kernel
void memcopy_kernel(__global T *out, dims_t ostrides,
                    __global const T *in, dims_t idims,
                    dims_t istrides, int offset,
                    int groups_0, int groups_1)
{
    const int lid0 = get_local_id(0);
    const int lid1 = get_local_id(1);

    const int id2 = get_group_id(0) / groups_0;
    const int id3 = get_group_id(1) / groups_1;
    const int group_id_0 = get_group_id(0) - groups_0 * id2;
    const int group_id_1 = get_group_id(1) - groups_1 * id3;
    const int id0 = group_id_0 * get_local_size(0) + lid0;
    const int id1 = group_id_1 * get_local_size(1) + lid1;

    in += offset;

    // FIXME: Do more work per work group
    out += id3 * ostrides.dim[3] + id2 * ostrides.dim[2] + id1 * ostrides.dim[1];
    in  += id3 * istrides.dim[3] + id2 * istrides.dim[2] + id1 * istrides.dim[1];

    int istride0 = istrides.dim[0];
    if (id0 < idims.dim[0] &&
        id1 < idims.dim[1] &&
        id2 < idims.dim[2] &&
        id3 < idims.dim[3]) {
        out[id0] = in[id0 * istride0];
    }
}
