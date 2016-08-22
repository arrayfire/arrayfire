/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 *
 ********************************************************/

__kernel void initState(__global uint *state, __global uint *tbl, ulong seed)
{
    __local uint lstate[N];
    const __global uint *ltbl = tbl + (TABLE_SIZE*get_group_id(0));
    uint hidden_seed = ltbl[4] ^ (ltbl[8] << 16);
    uint tmp = hidden_seed;
    tmp += tmp >> 16;
    tmp += tmp >> 8;
    tmp &= 0xff;
    tmp |= tmp << 8;
    tmp |= tmp << 16;
    lstate[get_local_id(0)] = tmp;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) == 0) {
        lstate[0] = seed;
        lstate[1] = hidden_seed;
        for (int i = 1; i < N; ++i) {
            lstate[i] ^= (uint)(1812433253) * (lstate[i-1] ^ (lstate[i-1] >> 30)) + i;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    state[N*get_group_id(0) + get_local_id(0)] = lstate[get_local_id(0)];
}

