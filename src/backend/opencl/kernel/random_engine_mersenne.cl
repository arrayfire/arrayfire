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
#define STATE_SIZE (256 * 3)

#define divup(NUM, DEN) (((NUM) + (DEN)-1) / (DEN));

void read_table(local uint *const localTable, global const uint *const table) {
    global const uint *const t = table + (get_group_id(0) * TABLE_SIZE);
    if (get_local_id(0) < TABLE_SIZE) {
        localTable[get_local_id(0)] = t[get_local_id(0)];
    }
}

void state_read(local uint *const localState, global const uint *const state) {
    global const uint *const g = state + (get_group_id(0) * N);
    localState[STATE_SIZE - N + get_local_id(0)] = g[get_local_id(0)];
    if (get_local_id(0) < N - THREADS) {
        localState[STATE_SIZE - N + THREADS + get_local_id(0)] =
            g[THREADS + get_local_id(0)];
    }
}

void state_write(global uint *const state, local const uint *const localState) {
    global uint *const g = state + (get_group_id(0) * N);
    g[get_local_id(0)]   = localState[STATE_SIZE - N + get_local_id(0)];
    if (get_local_id(0) < N - THREADS) {
        g[THREADS + get_local_id(0)] =
            localState[STATE_SIZE - N + THREADS + get_local_id(0)];
    }
}

uint recursion(local const uint *const recursion_table, const uint mask,
               const uint sh1, const uint sh2, const uint x1, const uint x2,
               uint y) {
    uint x = (x1 & mask) ^ x2;
    x ^= x << sh1;
    y        = x ^ (y >> sh2);
    uint mat = recursion_table[y & 0x0f];
    return y ^ mat;
}

uint temper(local const uint *const temper_table, const uint v, uint t) {
    t ^= t >> 16;
    t ^= t >> 8;
    uint mat = temper_table[t & 0x0f];
    return v ^ mat;
}

kernel void mersenneGenerator(global T *output, global uint *const state,
                              global const uint *const pos_tbl,
                              global const uint *const sh1_tbl,
                              global const uint *const sh2_tbl, uint mask,
                              global const uint *const recursion_table,
                              global const uint *const temper_table,
                              uint elements_per_block, uint elements) {
    local uint l_state[STATE_SIZE];
    local uint l_recursion_table[TABLE_SIZE];
    local uint l_temper_table[TABLE_SIZE];
    uint start = get_group_id(0) * elements_per_block;
    uint end   = start + elements_per_block;
    end        = (end > elements) ? elements : end;
    int iter   = divup((end - start) * sizeof(T), THREADS * 4 * sizeof(uint));
    uint pos   = pos_tbl[get_group_id(0)];
    uint sh1   = sh1_tbl[get_group_id(0)];
    uint sh2   = sh2_tbl[get_group_id(0)];

    state_read(l_state, state);
    read_table(l_recursion_table, recursion_table);
    read_table(l_temper_table, temper_table);
    barrier(CLK_LOCAL_MEM_FENCE);

    uint index                    = start;
    int elementsPerBlockIteration = THREADS * 4 * sizeof(uint) / sizeof(T);
    uint o[4];
    int offsetX1 = (STATE_SIZE - N + get_local_id(0)) % STATE_SIZE;
    int offsetX2 = (STATE_SIZE - N + get_local_id(0) + 1) % STATE_SIZE;
    int offsetY  = (STATE_SIZE - N + get_local_id(0) + pos) % STATE_SIZE;
    int offsetT  = (STATE_SIZE - N + get_local_id(0) + pos - 1) % STATE_SIZE;
    int offsetO  = get_local_id(0);

    for (int i = 0; i < iter; ++i) {
        for (int ii = 0; ii < 4; ++ii) {
            uint r =
                recursion(l_recursion_table, mask, sh1, sh2, l_state[offsetX1],
                          l_state[offsetX2], l_state[offsetY]);
            l_state[offsetO] = r;
            o[ii]            = temper(l_temper_table, r, l_state[offsetT]);
            offsetX1 += THREADS;
            offsetX2 += THREADS;
            offsetY += THREADS;
            offsetT += THREADS;
            offsetO += THREADS;
            offsetX1 =
                (offsetX1 >= STATE_SIZE) ? offsetX1 - STATE_SIZE : offsetX1;
            offsetX2 =
                (offsetX2 >= STATE_SIZE) ? offsetX2 - STATE_SIZE : offsetX2;
            offsetY = (offsetY >= STATE_SIZE) ? offsetY - STATE_SIZE : offsetY;
            offsetT = (offsetT >= STATE_SIZE) ? offsetT - STATE_SIZE : offsetT;
            offsetO = (offsetO >= STATE_SIZE) ? offsetO - STATE_SIZE : offsetO;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        uint writeIndex = index + get_local_id(0);
        if (i == iter - 1) {
            PARTIAL_WRITE(output, writeIndex, o[0], o[1], o[2], o[3], elements);
        } else {
            WRITE(output, writeIndex, o[0], o[1], o[2], o[3]);
        }
        index += elementsPerBlockIteration;
    }
    state_write(state, l_state);
}
