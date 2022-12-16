/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <math.hpp>
#include <shared.hpp>

namespace arrayfire {
namespace cuda {

inline __device__ int max_val(const int x, const int y) { return max(x, y); }
inline __device__ unsigned max_val(const unsigned x, const unsigned y) {
    return max(x, y);
}
inline __device__ float max_val(const float x, const float y) {
    return fmax(x, y);
}
inline __device__ double max_val(const double x, const double y) {
    return fmax(x, y);
}

template<typename T>
__global__ void susan(T* out, const T* in, const unsigned idim0,
                      const unsigned idim1, const unsigned radius,
                      const float t, const float g, const unsigned edge) {
    const int rSqrd   = radius * radius;
    const int windLen = 2 * radius + 1;
    const int shrdLen = BLOCK_X + windLen - 1;

    SharedMemory<T> shared;
    T* shrdMem = shared.getPointer();

    const unsigned lx = threadIdx.x;
    const unsigned ly = threadIdx.y;
    const unsigned gx = blockDim.x * blockIdx.x + lx + edge;
    const unsigned gy = blockDim.y * blockIdx.y + ly + edge;

    const unsigned nucleusIdx = (ly + radius) * shrdLen + lx + radius;
    shrdMem[nucleusIdx] = gx < idim0 && gy < idim1 ? in[gy * idim0 + gx] : 0;
    T m_0               = shrdMem[nucleusIdx];

#pragma unroll
    for (int b = ly, gy2 = gy; b < shrdLen; b += BLOCK_Y, gy2 += BLOCK_Y) {
        int j = gy2 - radius;
#pragma unroll
        for (int a = lx, gx2 = gx; a < shrdLen; a += BLOCK_X, gx2 += BLOCK_X) {
            int i = gx2 - radius;
            shrdMem[b * shrdLen + a] =
                (i < idim0 && j < idim1 ? in[j * idim0 + i] : m_0);
        }
    }
    __syncthreads();

    if (gx < idim0 - edge && gy < idim1 - edge) {
        unsigned idx = gy * idim0 + gx;
        float nM     = 0.0f;
#pragma unroll
        for (int p = 0; p < windLen; ++p) {
#pragma unroll
            for (int q = 0; q < windLen; ++q) {
                int i = p - radius;
                int j = q - radius;
                int a = lx + radius + i;
                int b = ly + radius + j;
                if (i * i + j * j < rSqrd) {
                    float c       = m_0;
                    float m       = shrdMem[b * shrdLen + a];
                    float exp_pow = powf((m - c) / t, 6.0f);
                    float cM      = expf(-exp_pow);
                    nM += cM;
                }
            }
        }
        out[idx] = nM < g ? g - nM : T(0);
    }
}

template<typename T>
__global__ void nonMax(float* x_out, float* y_out, float* resp_out,
                       unsigned* count, const unsigned idim0,
                       const unsigned idim1, const T* resp_in,
                       const unsigned edge, const unsigned max_corners) {
    // Responses on the border don't have 8-neighbors to compare, discard them
    const unsigned r = edge + 1;

    const unsigned gx = blockDim.x * blockIdx.x + threadIdx.x + r;
    const unsigned gy = blockDim.y * blockIdx.y + threadIdx.y + r;

    if (gx < idim0 - r && gy < idim1 - r) {
        const T v = resp_in[gy * idim0 + gx];

        // Find maximum neighborhood response
        T max_v;
        max_v = max_val(resp_in[(gy - 1) * idim0 + gx - 1],
                        resp_in[gy * idim0 + gx - 1]);
        max_v = max_val(max_v, resp_in[(gy + 1) * idim0 + gx - 1]);
        max_v = max_val(max_v, resp_in[(gy - 1) * idim0 + gx]);
        max_v = max_val(max_v, resp_in[(gy + 1) * idim0 + gx]);
        max_v = max_val(max_v, resp_in[(gy - 1) * idim0 + gx + 1]);
        max_v = max_val(max_v, resp_in[(gy)*idim0 + gx + 1]);
        max_v = max_val(max_v, resp_in[(gy + 1) * idim0 + gx + 1]);

        // Stores corner to {x,y,resp}_out if it's response is maximum compared
        // to its 8-neighborhood and greater or equal minimum response
        if (v > max_v) {
            unsigned idx = atomicAdd(count, 1u);
            if (idx < max_corners) {
                x_out[idx]    = (float)gx;
                y_out[idx]    = (float)gy;
                resp_out[idx] = (float)v;
            }
        }
    }
}

}  // namespace cuda
}  // namespace arrayfire
