/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

__kernel
void histogram(__global outType *         d_dst,
               KParam                     oInfo,
               __global const inType *    d_src,
               KParam                     iInfo,
               __global const float2 *    d_minmax,
               __local outType *          localMem,
               dim_type len, dim_type nbins, dim_type blk_x)
{
    // offset minmax array to account for batch ops
    __global const float2 * d_mnmx = d_minmax + get_group_id(1);

    // offset input and output to account for batch ops
    __global const inType *in = d_src + get_group_id(1) * iInfo.strides[2] + iInfo.offset;
    __global outType * out    = d_dst + get_group_id(1) * oInfo.strides[2];

    dim_type start = get_group_id(0) * THRD_LOAD * get_local_size(0) + get_local_id(0);
    dim_type end   = min((dim_type)(start + THRD_LOAD * get_local_size(0)), len);

    __local float minval;
    __local float dx;

    if (get_local_id(0) == 0) {
        float2 minmax = *d_mnmx;
        minval = minmax.s0;
        dx     = (minmax.s1-minmax.s0) / (float)nbins;
    }

    for (dim_type i = get_local_id(0); i < nbins; i += get_local_size(0))
        localMem[i] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int row = start; row < end; row += get_local_size(0)) {
        int bin = (int)(((float)in[row] - minval) / dx);
        bin     = max(bin, 0);
        bin     = min(bin, (int)nbins-1);
        atomic_inc((localMem + bin));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (dim_type i = get_local_id(0); i < nbins; i += get_local_size(0)) {
        atomic_add((out + i), localMem[i]);
    }
}
