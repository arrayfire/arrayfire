/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

/********************************************************
 * Copyright (c) 2009, 2010 Mutsuo Saito, Makoto Matsumoto and Hiroshima
 * University.
 * Copyright (c) 2011, 2012 Mutsuo Saito, Makoto Matsumoto, Hiroshima
 * University and University of Tokyo.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *     * Neither the name of the Hiroshima University, The Uinversity
 *       of Tokyo nor the names of its contributors may be used to
 *       endorse or promote products derived from this software without
 *       specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *******************************************************/

#define N 351
#define TABLE_SIZE 16

kernel void mersenneInitState(global uint *state, global uint *tbl,
                              ulong seed) {
    int tid      = get_local_id(0);
    int nthreads = get_local_size(0);
    int gid      = get_group_id(0);
    local uint lstate[N];
    const global uint *ltbl = tbl + (TABLE_SIZE * gid);
    uint hidden_seed        = ltbl[4] ^ (ltbl[8] << 16);
    uint tmp                = hidden_seed;
    tmp += tmp >> 16;
    tmp += tmp >> 8;
    tmp &= 0xff;
    tmp |= tmp << 8;
    tmp |= tmp << 16;

    for (int id = tid; id < N; id += nthreads) { lstate[id] = tmp; }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid == 0) {
        lstate[0] = seed;
        lstate[1] = hidden_seed;
        for (int i = 1; i < N; ++i) {
            lstate[i] ^=
                (uint)(1812433253) * (lstate[i - 1] ^ (lstate[i - 1] >> 30)) +
                i;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int id = tid; id < N; id += nthreads) {
        state[N * gid + id] = lstate[id];
    }
}
