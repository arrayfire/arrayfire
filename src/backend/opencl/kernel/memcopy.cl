/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

typedef struct {
    dim_type dim[4];
} dims_t;

__kernel
void memcopy_kernel(__global T *out, dims_t ostrides,
                    __global const T *in, dims_t idims,
                    dims_t istrides, dim_type offset,
                    dim_type groups_0, dim_type groups_1)
{
    const dim_type lid0 = get_local_id(0);
    const dim_type lid1 = get_local_id(1);

    const dim_type id2 = get_group_id(0) / groups_0;
    const dim_type id3 = get_group_id(1) / groups_1;
    const dim_type group_id_0 = get_group_id(0) - groups_0 * id2;
    const dim_type group_id_1 = get_group_id(1) - groups_1 * id3;
    const dim_type id0 = group_id_0 * get_local_size(0) + lid0;
    const dim_type id1 = group_id_1 * get_local_size(1) + lid1;

    in += offset;

    // FIXME: Do more work per work group
    out += id3 * ostrides.dim[3] + id2 * ostrides.dim[2] + id1 * ostrides.dim[1];
    in  += id3 * istrides.dim[3] + id2 * istrides.dim[2] + id1 * istrides.dim[1];

    dim_type istride0 = istrides.dim[0];
    if (id0 < idims.dim[0] &&
        id1 < idims.dim[1] &&
        id2 < idims.dim[2] &&
        id3 < idims.dim[3]) {
        out[id0] = in[id0 * istride0];
    }
}
