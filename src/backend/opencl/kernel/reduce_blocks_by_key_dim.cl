/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// Starting from OpenCL 2.0, core profile includes work group level
// inclusive scan operations, hence skip defining custom one
#if __OPENCL_C_VERSION__ == 200 || __OPENCL_C_VERSION__ == 210 || \
    __OPENCL_C_VERSION__ == 220 || __opencl_c_work_group_collective_functions
#define BUILTIN_WORK_GROUP_COLLECTIVE_FUNCTIONS
#endif

#ifndef BUILTIN_WORK_GROUP_COLLECTIVE_FUNCTIONS
int work_group_scan_inclusive_add(local int *wg_temp, __local int *arr) {
    local int *active_buf;

    const int lid = get_local_id(0);
    int val       = arr[lid];
    active_buf    = arr;

    bool swap_buffer = false;
    for (int off = 1; off <= DIMX; off *= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid >= off) { val = val + active_buf[lid - off]; }
        swap_buffer     = !swap_buffer;
        active_buf      = swap_buffer ? wg_temp : arr;
        active_buf[lid] = val;
    }

    int res = active_buf[lid];
    return res;
}
#endif

kernel void reduce_blocks_by_key_dim(global int *reduced_block_sizes,
                                     global Tk *oKeys, KParam oKInfo,
                                     global To *oVals, KParam oVInfo,
                                     const global Tk *iKeys, KParam iKInfo,
                                     const global Ti *iVals, KParam iVInfo,
                                     int change_nan, To nanval, int n,
                                     const int nBlocksZ) {
    const uint lid  = get_local_id(0);
    const uint gidx = get_global_id(0);

    const int bidy = get_group_id(1);
    const int bidz = get_group_id(2) % nBlocksZ;
    const int bidw = get_group_id(2) / nBlocksZ;

    local Tk keys[DIMX];
    local To vals[DIMX];
    local Tk reduced_keys[DIMX];
    local To reduced_vals[DIMX];
    local int unique_ids[DIMX];
#ifndef BUILTIN_WORK_GROUP_COLLECTIVE_FUNCTIONS
    local int wg_temp[DIMX];
    local int unique_flags[DIMX];
#endif

    const To init_val = init;

    //
    // will hold final number of reduced elements in block
    local int reducedBlockSize;

    local int dims_ordering[4];
    if (lid == 0) {
        reducedBlockSize = 0;

        int d            = 1;
        dims_ordering[0] = DIM;
        for (int i = 0; i < 4; ++i) {
            if (i != DIM) dims_ordering[d++] = i;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // load keys and values to threads
    Tk k;
    To v;
    if (gidx < n) {
        k             = iKeys[gidx];
        const int gid = bidw * iVInfo.strides[dims_ordering[3]] +
                        bidz * iVInfo.strides[dims_ordering[2]] +
                        bidy * iVInfo.strides[dims_ordering[1]] +
                        gidx * iVInfo.strides[DIM];
        v = transform(iVals[gid]);
        if (change_nan) v = IS_NAN(v) ? nanval : v;
    } else {
        v = init_val;
    }

    keys[lid] = k;
    vals[lid] = v;

    reduced_keys[lid] = k;
    barrier(CLK_LOCAL_MEM_FENCE);

    // mark threads containing unique keys
    int eq_check    = (lid > 0) ? (k != reduced_keys[lid - 1]) : 0;
    int unique_flag = (eq_check || (lid == 0)) && (gidx < n);

#ifdef BUILTIN_WORK_GROUP_COLLECTIVE_FUNCTIONS
    int unique_id = work_group_scan_inclusive_add(unique_flag);
#else
    unique_flags[lid] = unique_flag;
    int unique_id     = work_group_scan_inclusive_add(wg_temp, unique_flags);
#endif
    unique_ids[lid] = unique_id;

    if (lid == DIMX - 1) reducedBlockSize = unique_id;

    for (int off = 1; off < DIMX; off *= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        int test_unique_id =
            (lid + off < DIMX) ? unique_ids[lid + off] : ~unique_id;
        eq_check = (unique_id == test_unique_id);
        int update_key =
            eq_check && (lid < (DIMX - off)) &&
            ((gidx + off) <
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
        const int bOffset = bidw * oVInfo.strides[dims_ordering[3]] +
                            bidz * oVInfo.strides[dims_ordering[2]] +
                            bidy * oVInfo.strides[dims_ordering[1]];
        oKeys[gidx]                                 = reduced_keys[lid];
        oVals[bOffset + (gidx)*oVInfo.strides[DIM]] = reduced_vals[lid];
    }

    reduced_block_sizes[bid] = reducedBlockSize;
}
