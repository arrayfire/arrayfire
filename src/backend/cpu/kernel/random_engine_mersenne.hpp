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

#pragma once

namespace arrayfire {
namespace cpu {
namespace kernel {

static const int N          = 351;
static const int STATE_SIZE = 256 * 3;

uint recursion(const uint* const recursion_table, const uint mask,
               const uint sh1, const uint sh2, const uint x1, const uint x2,
               uint y) {
    uint x = (x1 & mask) ^ x2;
    x ^= x << sh1;
    y        = x ^ (y >> sh2);
    uint mat = recursion_table[y & 0x0f];
    return y ^ mat;
}

uint temper(const uint* const temper_table, const uint v, uint t) {
    t ^= t >> 16;
    t ^= t >> 8;
    uint mat = temper_table[t & 0x0f];
    return v ^ mat;
}

void mersenne(uint* const out, uint* const state, int i, uint pos, uint sh1,
              uint sh2, uint mask, const uint* const recursion_table,
              const uint* const temper_table) {
    int index    = i % STATE_SIZE;
    int offsetX1 = (STATE_SIZE - N + index) % STATE_SIZE;
    int offsetX2 = (STATE_SIZE - N + index + 1) % STATE_SIZE;
    int offsetY  = (STATE_SIZE - N + index + pos) % STATE_SIZE;
    int offsetT  = (STATE_SIZE - N + index + pos - 1) % STATE_SIZE;
    for (int i = 0; i < 4; ++i) {
        state[index] =
            recursion(recursion_table, mask, sh1, sh2, state[offsetX1],
                      state[offsetX2], state[offsetY]);
        out[i]   = temper(temper_table, state[index], state[offsetT]);
        offsetX1 = (offsetX1 + 1) % STATE_SIZE;
        offsetX2 = (offsetX2 + 1) % STATE_SIZE;
        offsetY  = (offsetY + 1) % STATE_SIZE;
        offsetT  = (offsetT + 1) % STATE_SIZE;
        index    = (index + 1) % STATE_SIZE;
    }
}

void state_read(uint* const l_state, const uint* const state) {
    for (int i = 0; i < N; ++i) { l_state[STATE_SIZE - N + i] = state[i]; }
}

void state_write(uint* const state, const uint* const l_state) {
    for (int i = 0; i < N; ++i) { state[i] = l_state[STATE_SIZE - N + i]; }
}

void initMersenneState(uint* const state, const uint* const tbl,
                       const uintl seed) {
    uint hidden_seed = tbl[4] ^ (tbl[8] << 16);
    uint tmp         = hidden_seed;
    tmp += tmp >> 16;
    tmp += tmp >> 8;
    tmp &= 0xff;
    tmp |= tmp << 8;
    tmp |= tmp << 16;
    state[0] = seed;
    state[1] =
        hidden_seed ^ ((uint)(1812433253) * (state[0] ^ (state[0] >> 30)) + 1);
    for (int i = 2; i < N; ++i) {
        state[i] = tmp;
        state[i] ^=
            (uint)(1812433253) * (state[i - 1] ^ (state[i - 1] >> 30)) + i;
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
