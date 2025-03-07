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

namespace arrayfire {
namespace cuda {
namespace kernel {

constexpr int N          = 351;
constexpr int BLOCKS     = 32;
constexpr int STATE_SIZE = (256 * 3);
constexpr int TABLE_SIZE = 16;

// Utils
static inline __device__ void read_table(uint *const sharedTable,
                                         const uint *const table) {
    const uint *const t = table + (blockIdx.x * TABLE_SIZE);
    if (threadIdx.x < TABLE_SIZE) { sharedTable[threadIdx.x] = t[threadIdx.x]; }
}

static inline __device__ void state_read(uint *const state,
                                         const uint *const gState) {
    const uint *const g                 = gState + (blockIdx.x * N);
    state[STATE_SIZE - N + threadIdx.x] = g[threadIdx.x];
    if (threadIdx.x < N - blockDim.x) {
        state[STATE_SIZE - N + blockDim.x + threadIdx.x] =
            g[blockDim.x + threadIdx.x];
    }
}

static inline __device__ void state_write(uint *const gState,
                                          const uint *const state) {
    uint *const g  = gState + (blockIdx.x * N);
    g[threadIdx.x] = state[STATE_SIZE - N + threadIdx.x];
    if (threadIdx.x < N - blockDim.x) {
        g[blockDim.x + threadIdx.x] =
            state[STATE_SIZE - N + blockDim.x + threadIdx.x];
    }
}

static inline __device__ uint recursion(const uint *const recursion_table,
                                        const uint mask, const uint sh1,
                                        const uint sh2, const uint x1,
                                        const uint x2, uint y) {
    uint x = (x1 & mask) ^ x2;
    x ^= x << sh1;
    y        = x ^ (y >> sh2);
    uint mat = recursion_table[y & 0x0f];
    return y ^ mat;
}

static inline __device__ uint temper(const uint *const temper_table,
                                     const uint v, uint t) {
    t ^= t >> 16;
    t ^= t >> 8;
    uint mat = temper_table[t & 0x0f];
    return v ^ mat;
}

// Initialization

__global__ void initState(uint *state, const uint *tbl, uintl seed) {
    __shared__ uint lstate[N];
    const uint *ltbl = tbl + (TABLE_SIZE * blockIdx.x);
    uint hidden_seed = ltbl[4] ^ (ltbl[8] << 16);
    uint tmp         = hidden_seed;
    tmp += tmp >> 16;
    tmp += tmp >> 8;
    tmp &= 0xff;
    tmp |= tmp << 8;
    tmp |= tmp << 16;
    lstate[threadIdx.x] = tmp;
    __syncthreads();
    if (threadIdx.x == 0) {
        lstate[0] = seed;
        lstate[1] = hidden_seed;
        for (int i = 1; i < N; ++i) {
            lstate[i] ^=
                ((uint)(1812433253) * (lstate[i - 1] ^ (lstate[i - 1] >> 30)) +
                 i);
        }
    }
    __syncthreads();
    state[N * blockIdx.x + threadIdx.x] = lstate[threadIdx.x];
}

void initMersenneState(uint *state, const uint *tbl, uintl seed) {
    CUDA_LAUNCH(initState, BLOCKS, N, state, tbl, seed);
}
}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
