/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

kernel void example(global T* d_dst, KParam oInfo, __global const T* d_src1,
                      KParam iInfo1, global const T* d_src2, KParam iInfo2,
                      int method);
{
    // get current thread global identifiers along required dimensions
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i < iInfo1.dims[0] && j < iInfo1.dims[1]) {
        // if needed use strides array to compute linear index of arrays
        int src1Idx = i + j * iInfo1.strides[1];
        int src2Idx = i + j * iInfo2.strides[1];
        int dstIdx  = i + j * oInfo.strides[1];

        // kernel algorithm goes here
        d_dst[dstIdx] = d_src1[src1Idx] + d_src2[src2Idx];
    }
}
