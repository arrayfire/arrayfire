/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

kernel void final_boundary_reduce_dim(global int *reduced_block_sizes,
                                      global Tk *oKeys, KParam oKInfo,
                                      global To *oVals, KParam oVInfo,
                                      const int n, const int nBlocksZ) {
    local int dim_ordering[4];

    const uint lid = get_local_id(0);
    const uint bid = get_group_id(0);
    const uint gid = get_global_id(0);

    const int bidy = get_group_id(1);
    const int bidz = get_group_id(2) % nBlocksZ;
    const int bidw = get_group_id(2) / nBlocksZ;

    if (lid == 0) {
        int d           = 1;
        dim_ordering[0] = DIM;
        for (int i = 0; i < 4; ++i) {
            if (i != DIM) dim_ordering[d++] = i;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid == ((bid + 1) * get_local_size(0)) - 1 &&
        bid < get_num_groups(0) - 1) {
        Tk k0 = oKeys[gid];
        Tk k1 = oKeys[gid + 1];
        if (k0 == k1) {
            To v0                    = oVals[gid];
            To v1                    = oVals[gid + 1];
            oVals[gid + 1]           = binOp(v0, v1);
            reduced_block_sizes[bid] = get_local_size(0) - 1;
        } else {
            reduced_block_sizes[bid] = get_local_size(0);
        }
    }

    // if last block, set block size to difference between n and block boundary
    if (lid == 0 && bid == get_num_groups(0) - 1) {
        reduced_block_sizes[bid] = n - (bid * get_local_size(0));
    }
}
