/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

namespace cpu
{
namespace kernel
{

    #define N 351
    #define STATE_SIZE 786

    uint recursion(const uint * const recursion_table, const uint mask,
            const uint sh1, const uint sh2, const uint x1, const uint x2, uint y)
    {
        uint x = (x1 & mask) ^ x2;
        x ^= x << sh1;
        y = x ^ (y >> sh2);
        uint mat = recursion_table[y & 0x0f];
        return y ^ mat;
    }

    uint temper(const uint * const temper_table, const uint v, uint t)
    {
         t ^= t >> 16;
         t ^= t >> 8;
         uint mat = temper_table[t & 0x0f];
         return v ^ mat;
    }

    void mersenne(uint * const out,
            uint * const state,
            int i,
            uint pos,
            uint sh1,
            uint sh2,
            uint mask,
            uint const * const recursion_table,
            uint const * const temper_table)
    {
        int index    = i % STATE_SIZE;
        int offsetX1 = (STATE_SIZE - N + index          ) % STATE_SIZE;
        int offsetX2 = (STATE_SIZE - N + index + 1      ) % STATE_SIZE;
        int offsetY  = (STATE_SIZE - N + index + pos    ) % STATE_SIZE;
        int offsetT  = (STATE_SIZE - N + index + pos - 1) % STATE_SIZE;
        for (int i = 0; i < 4; ++i) {
            state[index] = recursion(recursion_table, mask, sh1, sh2,
                    state[offsetX1], state[offsetX2], state[offsetY]);
            out[i] = temper(temper_table, state[index], state[offsetT]);
            offsetX1 = (offsetX1 + 1) % STATE_SIZE;
            offsetX2 = (offsetX2 + 1) % STATE_SIZE;
            offsetY  = (offsetY  + 1) % STATE_SIZE;
            offsetT  = (offsetT  + 1) % STATE_SIZE;
            index    = (index    + 1) % STATE_SIZE;
        }
    }

    void state_read(uint * const l_state, const uint * const state)
    {
        for (int i = 0; i < N; ++i) {
            l_state[STATE_SIZE - N + i] = state[i];
        }
    }

    void state_write(uint * const state, const uint * const l_state)
    {
        for (int i = 0; i < N; ++i) {
            state[i] = l_state[STATE_SIZE - N + i];
        }
    }

}
}
