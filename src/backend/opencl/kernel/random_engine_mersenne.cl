/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 *
 ********************************************************/

#define divup(N, D) (((N) + (D) - 1)/(D));

void read_table(__local uint * const localTable, __global const uint * const table)
{
    __global const uint * const t = table + (get_group_id(0) * TABLE_SIZE);
    if (get_local_id(0) < TABLE_SIZE) {
        localTable[get_local_id(0)] = t[get_local_id(0)];
    }
}

void state_read(__local uint * const localState, __global const uint * const state)
{
    __global const uint * const g = state + (get_group_id(0) * N);
    localState[STATE_SIZE - N + get_local_id(0)] = g[get_local_id(0)];
    if (get_local_id(0) < N - THREADS) {
        localState[STATE_SIZE - N + THREADS + get_local_id(0)] = g[THREADS + get_local_id(0)];
    }
}

void state_write(__global uint * const state, __local const uint * const localState)
{
    __global uint * const g = state + (get_group_id(0) * N);
    g[get_local_id(0)] = localState[STATE_SIZE - N + get_local_id(0)];
    if (get_local_id(0) < N - THREADS) {
        g[THREADS + get_local_id(0)] = localState[STATE_SIZE - N + THREADS + get_local_id(0)];
    }
}

uint recursion(__local const uint * const recursion_table, const uint mask,
        const uint sh1, const uint sh2, const uint x1, const uint x2, uint y)
{
    uint x = (x1 & mask) ^ x2;
    x ^= x << sh1;
    y = x ^ (y >> sh2);
    uint mat = recursion_table[y & 0x0f];
    return y ^ mat;
}

uint temper(__local const uint * const temper_table, const uint v, uint t)
{
     t ^= t >> 16;
     t ^= t >> 8;
     uint mat = temper_table[t & 0x0f];
     return v ^ mat;
}

__kernel void generate(__global T *output,
        __global uint * const state,
        __global const uint * const pos_tbl,
        __global const uint * const sh1_tbl,
        __global const uint * const sh2_tbl,
        uint mask,
        __global const uint * const recursion_table,
        __global const uint * const temper_table,
        uint elements_per_block, uint elements)
{
    __local uint l_state[STATE_SIZE];
    __local uint l_recursion_table[TABLE_SIZE];
    __local uint l_temper_table[TABLE_SIZE];
    uint start = get_group_id(0)*elements_per_block;
    uint end = start + elements_per_block;
    end = (end > elements)? elements : end;
    int iter = divup((end - start)*sizeof(T), THREADS*4*sizeof(uint));
    uint pos = pos_tbl[get_group_id(0)];
    uint sh1 = sh1_tbl[get_group_id(0)];
    uint sh2 = sh2_tbl[get_group_id(0)];

    state_read(l_state, state);
    read_table(l_recursion_table, recursion_table);
    read_table(l_temper_table, temper_table);
    barrier(CLK_LOCAL_MEM_FENCE);

    uint index = start;
    int elementsPerBlockIteration = THREADS*4*sizeof(uint)/sizeof(T);
    uint o[4];
    int offsetX1 = (STATE_SIZE - N + get_local_id(0)          ) % STATE_SIZE;
    int offsetX2 = (STATE_SIZE - N + get_local_id(0) + 1      ) % STATE_SIZE;
    int offsetY  = (STATE_SIZE - N + get_local_id(0) + pos    ) % STATE_SIZE;
    int offsetT  = (STATE_SIZE - N + get_local_id(0) + pos - 1) % STATE_SIZE;
    int offsetO  = get_local_id(0);

    for (int i = 0; i < iter; ++i) {
        for (int ii = 0; ii < 4; ++ii) {
            uint r = recursion(l_recursion_table, mask, sh1, sh2,
                    l_state[offsetX1],
                    l_state[offsetX2],
                    l_state[offsetY ]);
            l_state[offsetO] = r;
            o[ii] = temper(l_temper_table, r, l_state[offsetT]);
            offsetX1 = (offsetX1 + THREADS) % STATE_SIZE;
            offsetX2 = (offsetX2 + THREADS) % STATE_SIZE;
            offsetY  = (offsetY  + THREADS) % STATE_SIZE;
            offsetT  = (offsetT  + THREADS) % STATE_SIZE;
            offsetO  = (offsetO  + THREADS) % STATE_SIZE;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        uint writeIndex = index + get_local_id(0);
        if (i == iter - 1) {
            PARTIAL_WRITE(output, &writeIndex, &o[0], &o[1], &o[2], &o[3], &elements);
        } else {
            WRITE(output, &writeIndex, &o[0], &o[1], &o[2], &o[3]);
        }
        index += elementsPerBlockIteration;
    }
    state_write(state, l_state);
}

