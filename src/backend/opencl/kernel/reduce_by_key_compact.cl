/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

kernel void compact(global int *reduced_block_sizes, global Tk *oKeys,
                    KParam oKInfo, global To *oVals, KParam oVInfo,
                    const global Tk *iKeys, KParam iKInfo,
                    const global To *iVals, KParam iVInfo, const int nBlocksZ) {
    const uint lid = get_local_id(0);
    const uint bid = get_group_id(0);
    const uint gid = get_global_id(0);

    const int bidy = get_group_id(1);
    const int bidz = get_group_id(2) % nBlocksZ;
    const int bidw = get_group_id(2) / nBlocksZ;

    Tk k;
    To v;

    const int bOffset = bidw * oVInfo.strides[3] + bidz * oVInfo.strides[2] +
                        bidy * oVInfo.strides[1];

    // reduced_block_sizes should have inclusive sum of block sizes
    int nwrite =
        (bid == 0) ? reduced_block_sizes[0]
                   : (reduced_block_sizes[bid] - reduced_block_sizes[bid - 1]);
    int writeloc = (bid == 0) ? 0 : reduced_block_sizes[bid - 1];

    k = iKeys[gid];
    v = iVals[bOffset + gid];

    if (lid < nwrite) {
        oKeys[writeloc + lid]           = k;
        oVals[bOffset + writeloc + lid] = v;
    }
}
