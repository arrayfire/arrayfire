/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

kernel void test_needs_reduction(global int *needs_another_reduction,
                                   global int *needs_block_boundary_reduced,
                                   const global Tk *iKeys, KParam iKInfo,
                                   int n) {
    const uint lid = get_local_id(0);
    const uint bid = get_group_id(0);
    const uint gid = get_global_id(0);

    Tk k;
    if (gid < n) { k = iKeys[gid]; }

    local Tk keys[DIMX];
    keys[lid] = k;
    barrier(CLK_LOCAL_MEM_FENCE);

    int update_key =
        (lid < DIMX - 2) && (k == keys[lid + 1]) && (gid < (n - 1));

    if (update_key) { atomic_or(needs_another_reduction, update_key); }

    barrier(CLK_LOCAL_MEM_FENCE);

    // last thread in each block checks if any inter-block keys need further
    // reduction
    if (gid == ((bid + 1) * DIMX) - 1 && bid < get_num_groups(0) - 1) {
        int k0 = iKeys[gid];
        int k1 = iKeys[gid + 1];
        if (k0 == k1) { atomic_or(needs_block_boundary_reduced, 1); }
    }
}
