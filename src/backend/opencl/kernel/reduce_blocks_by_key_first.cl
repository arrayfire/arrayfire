/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

Tk work_group_scan_inclusive_add(__local Tk *arr) {
    __local Tk tmp[DIMX];
    __local int *l_val;

    const int lid = get_local_id(0);
    Tk val        = arr[lid];
    l_val         = arr;

    bool wbuf = 0;
    for (int off = 1; off <= DIMX; off *= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid >= off) val = val + l_val[lid - off];

        wbuf       = 1 - wbuf;
        l_val      = wbuf ? tmp : arr;
        l_val[lid] = val;
    }

    Tk res = l_val[lid];
    return res;
}

__kernel void reduce_blocks_by_key_first(
    __global int *reduced_block_sizes, __global Tk *oKeys, KParam oKInfo,
    __global To *oVals, KParam oVInfo, const __global Tk *iKeys, KParam iKInfo,
    const __global Ti *iVals, KParam iVInfo, int change_nan, To nanval, int n,
    const int nBlocksZ) {
    const uint lid = get_local_id(0);
    const uint gid = get_global_id(0);

    const int bidy = get_group_id(1);
    const int bidz = get_group_id(2) % nBlocksZ;
    const int bidw = get_group_id(2) / nBlocksZ;

    __local Tk keys[DIMX];
    __local To vals[DIMX];

    __local Tk reduced_keys[DIMX];
    __local To reduced_vals[DIMX];

    __local int unique_flags[DIMX];
    __local int unique_ids[DIMX];

    const To init_val = init;

    //
    // will hold final number of reduced elements in block
    __local int reducedBlockSize;

    if (lid == 0) { reducedBlockSize = 0; }

    // load keys and values to threads
    Tk k;
    To v;
    if (gid < n) {
        k                 = iKeys[gid];
        const int bOffset = bidw * iVInfo.strides[3] +
                            bidz * iVInfo.strides[2] + bidy * iVInfo.strides[1];
        v                 = transform(iVals[bOffset + gid]);
        if (change_nan) v = IS_NAN(v) ? nanval : v;
    } else {
        v = init_val;
    }


    keys[lid] = k;
    vals[lid] = v;

    reduced_keys[lid] = k;
    barrier(CLK_LOCAL_MEM_FENCE);

    // mark threads containing unique keys
    int eq_check      = (lid > 0) ? (k != reduced_keys[lid - 1]) : 0;
    int unique_flag   = (eq_check || (lid == 0)) && (gid < n);
    unique_flags[lid] = unique_flag;

    int unique_id   = work_group_scan_inclusive_add(unique_flags);
    unique_ids[lid] = unique_id;

    if (lid == DIMX - 1) reducedBlockSize = unique_id;

    for (int off = 1; off < DIMX; off *= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        int test_unique_id =
            (lid + off < DIMX) ? unique_ids[lid + off] : ~unique_id;
        eq_check = (unique_id == test_unique_id);
        int update_key =
            eq_check && (lid < (DIMX - off)) &&
            ((gid + off) <
             n);  // checks if this thread should perform a reduction
        To uval = (update_key) ? vals[lid + off] : init_val;
        barrier(CLK_LOCAL_MEM_FENCE);
        vals[lid] = binOp(vals[lid], uval);  // update if thread requires it
    }

    if (unique_flag) {
        reduced_keys[unique_id - 1] = k;
        reduced_vals[unique_id - 1] = vals[lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int bid = get_group_id(0);
    if (lid < reducedBlockSize) {
        const int bOffset = bidw * oVInfo.strides[3] +
                            bidz * oVInfo.strides[2] + bidy * oVInfo.strides[1];
        oKeys[bid * DIMX + lid]               = reduced_keys[lid];
        oVals[bOffset + ((bid * DIMX) + lid)] = reduced_vals[lid];
    }

    reduced_block_sizes[bid] = reducedBlockSize;
}
