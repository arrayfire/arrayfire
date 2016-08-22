/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

namespace cuda
{
namespace kernel
{
#define N 351
#define BLOCKS 32
#define STATE_SIZE 786
#define TABLE_SIZE 16

    //Utils
    static inline __device__ void read_table(uint * const sharedTable, const uint * const table)
    {
        const uint * const t = table + (blockIdx.x * TABLE_SIZE);
        if (threadIdx.x < TABLE_SIZE) {
            sharedTable[threadIdx.x] = t[threadIdx.x];
        }
    }

    static inline __device__ void state_read(uint * const state, const uint * const gState)
    {
        const uint * const g = gState + (blockIdx.x * N);
        state[STATE_SIZE - N + threadIdx.x] = g[threadIdx.x];
        if (threadIdx.x < N - blockDim.x) {
            state[STATE_SIZE - N + blockDim.x + threadIdx.x] = g[blockDim.x + threadIdx.x];
        }
    }

    static inline __device__ void state_write(uint * const gState, const uint * const state)
    {
        uint * const g = gState + (blockIdx.x * N);
        g[threadIdx.x] = state[STATE_SIZE - N + threadIdx.x];
        if (threadIdx.x < N - blockDim.x) {
            g[blockDim.x + threadIdx.x] = state[STATE_SIZE - N + blockDim.x + threadIdx.x];
        }
    }

    static inline __device__ uint recursion(uint const * const recursion_table,
            const uint mask, const uint sh1, const uint sh2,
            const uint x1, const uint x2, uint y)
    {
        uint x = (x1 & mask) ^ x2;
        x ^= x << sh1;
        y = x ^ (y >> sh2);
        uint mat = recursion_table[y & 0x0f];
        return y ^ mat;
    }

    static inline __device__ uint temper(const uint * const temper_table, const uint v, uint t)
    {
        t ^= t >> 16;
        t ^= t >> 8;
        uint mat = temper_table[t & 0x0f];
        return v ^ mat;
    }

    //Initialization

    __global__ void initState(uint *state, const uint *tbl, uintl seed)
    {
        __shared__ uint lstate[N];
        const uint *ltbl = tbl + (TABLE_SIZE*blockIdx.x);
        uint hidden_seed = ltbl[4] ^ (ltbl[8] << 16);
        uint tmp = hidden_seed;
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
                lstate[i] ^= (uint)(1812433253) * (lstate[i-1] ^ (lstate[i-1] >> 30)) + i;
            }
        }
        __syncthreads();
        state[N*blockIdx.x + threadIdx.x] = lstate[threadIdx.x];
    }

    void initMersenneState(uint *state, const uint *tbl, uintl seed)
    {
        CUDA_LAUNCH(initState, BLOCKS, N, state, tbl, seed);
    }
}
}
