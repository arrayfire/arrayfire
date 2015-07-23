/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <backend.hpp>
#include <Param.hpp>
#include <dispatch.hpp>
#include <debug_cuda.hpp>
#include <math.hpp>
#include "config.hpp"

namespace cuda
{

namespace kernel
{

static const unsigned BLOCK_X = 16;
static const unsigned BLOCK_Y = 16;

inline __device__ int max_val(const int x, const int y)
{
    return max(x, y);
}
inline __device__ unsigned max_val(const unsigned x, const unsigned y)
{
    return max(x, y);
}
inline __device__ float max_val(const float x, const float y)
{
    return fmax(x, y);
}
inline __device__ double max_val(const double x, const double y)
{
    return fmax(x, y);
}

template<typename T, unsigned radius>
__global__
void susanKernel(T* out, const T* in,
                 const unsigned idim0, const unsigned idim1,
                 const float t, const float g,
                 const unsigned edge)
{
    const int rSqrd   = radius*radius;
    const int windLen = 2*radius+1;
    const int shrdLen = BLOCK_X + windLen-1;
    const size_t SHRD_MEM_SIZE = (BLOCK_X+2*radius)*(BLOCK_Y+2*radius);
    __shared__ T shrdMem[SHRD_MEM_SIZE];

    const unsigned lx = threadIdx.x;
    const unsigned ly = threadIdx.y;
    const unsigned gx  = blockDim.x * blockIdx.x + lx + edge;
    const unsigned gy  = blockDim.y * blockIdx.y + ly + edge;

#pragma unroll
    for (int b=ly, gy2=gy; b<shrdLen; b+=BLOCK_Y, gy2+=BLOCK_Y) {
        int j = gy2-radius;
#pragma unroll
        for (int a=lx, gx2=gx; a<shrdLen; a+=BLOCK_X, gx2+=BLOCK_X) {
            int i = gx2-radius;
            shrdMem[b*shrdLen+a] = in[i*idim0+j];
        }
    }
    __syncthreads();

    if (gx < idim1 - edge && gy < idim0 - edge) {
        unsigned idx = gx*idim0 + gy;
        float nM  = 0.0f;
        float m_0 = in[idx];
#pragma unroll
        for (int p=0; p<windLen; ++p) {
#pragma unroll
            for (int q=0; q<windLen; ++q) {
                int i = p - radius;
                int j = q - radius;
                int a = lx + radius + i;
                int b = ly + radius + j;
                if (i*i + j*j < rSqrd) {
                    float m = shrdMem[b * shrdLen + a];
                    float exp_pow = powf((m - m_0)/t, 6.0f);
                    float cM = expf(-exp_pow);
                    nM += cM;
                }
            }
        }
        out[idx] = nM < g ? g - nM : T(0);
    }
}

template<typename T>
void susan_responses(T* out, const T* in,
                     const unsigned idim0, const unsigned idim1,
                     const int radius, const float t, const float g,
                     const unsigned edge)
{
    dim3 threads(BLOCK_X, BLOCK_Y);
    dim3 blocks(divup(idim1-edge*2, BLOCK_X), divup(idim0-edge*2, BLOCK_Y));

    switch (radius) {
        case 1: susanKernel<T, 1><<<blocks, threads>>>(out, in, idim0, idim1, t, g, edge); break;
        case 2: susanKernel<T, 2><<<blocks, threads>>>(out, in, idim0, idim1, t, g, edge); break;
        case 3: susanKernel<T, 3><<<blocks, threads>>>(out, in, idim0, idim1, t, g, edge); break;
        case 4: susanKernel<T, 4><<<blocks, threads>>>(out, in, idim0, idim1, t, g, edge); break;
        case 5: susanKernel<T, 5><<<blocks, threads>>>(out, in, idim0, idim1, t, g, edge); break;
        case 6: susanKernel<T, 6><<<blocks, threads>>>(out, in, idim0, idim1, t, g, edge); break;
        case 7: susanKernel<T, 7><<<blocks, threads>>>(out, in, idim0, idim1, t, g, edge); break;
        case 8: susanKernel<T, 8><<<blocks, threads>>>(out, in, idim0, idim1, t, g, edge); break;
        case 9: susanKernel<T, 9><<<blocks, threads>>>(out, in, idim0, idim1, t, g, edge); break;
    }

    POST_LAUNCH_CHECK();
}

template<typename T>
__global__
void nonMaxKernel(float* x_out, float* y_out, float* resp_out, unsigned* count,
                  const unsigned idim0, const unsigned idim1, const T* resp_in,
                  const unsigned edge, const unsigned max_corners)
{
    // Responses on the border don't have 8-neighbors to compare, discard them
    const unsigned r = edge + 1;

    const unsigned gx = blockDim.x * blockIdx.x + threadIdx.x + r;
    const unsigned gy = blockDim.y * blockIdx.y + threadIdx.y + r;

    if (gx < idim1 - r && gy < idim0 - r) {
        const T v = resp_in[gx * idim0 + gy];

        // Find maximum neighborhood response
        T max_v;
        max_v = max_val(resp_in[(gx-1) * idim0 + gy-1], resp_in[gx * idim0 + gy-1]);
        max_v = max_val(max_v, resp_in[(gx+1) * idim0 + gy-1]);
        max_v = max_val(max_v, resp_in[(gx-1) * idim0 + gy  ]);
        max_v = max_val(max_v, resp_in[(gx+1) * idim0 + gy  ]);
        max_v = max_val(max_v, resp_in[(gx-1) * idim0 + gy+1]);
        max_v = max_val(max_v, resp_in[(gx)   * idim0 + gy+1]);
        max_v = max_val(max_v, resp_in[(gx+1) * idim0 + gy+1]);

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

template<typename T>
void nonMaximal(float* x_out, float* y_out, float* resp_out,
                 unsigned* count, const unsigned idim0, const unsigned idim1,
                 const T * resp_in, const unsigned edge, const unsigned max_corners)
{
    dim3 threads(BLOCK_X, BLOCK_Y);
    dim3 blocks(divup(idim1-edge*2, BLOCK_X), divup(idim0-edge*2, BLOCK_Y));

    unsigned* d_corners_found = memAlloc<unsigned>(1);
    CUDA_CHECK(cudaMemset(d_corners_found, 0, sizeof(unsigned)));

    nonMaxKernel<T><<<blocks, threads>>>(x_out, y_out, resp_out, d_corners_found,
                                         idim0, idim1, resp_in, edge, max_corners);

    POST_LAUNCH_CHECK();

    CUDA_CHECK(cudaMemcpy(count, d_corners_found, sizeof(unsigned), cudaMemcpyDeviceToHost));
    memFree(d_corners_found);
}

__global__
void keepCornersKernel(float* x_out, float* y_out, float* resp_out,
                         const float* x_in, const float* y_in,
                         const float* resp_in, const unsigned* resp_idx,
                         const unsigned n_corners)
{
    const unsigned f = blockDim.x * blockIdx.x + threadIdx.x;

    // Keep only the first n_feat features
    if (f < n_corners) {
        x_out[f] = x_in[(unsigned)resp_idx[f]];
        y_out[f] = y_in[(unsigned)resp_idx[f]];
        resp_out[f] = resp_in[f];
    }
}

void keepCorners(float* x_out, float* y_out, float* resp_out,
                  const float* x_in, const float* y_in,
                  const float* resp_in, const unsigned* resp_idx,
                  const unsigned n_corners)
{
    dim3 threads(THREADS_PER_BLOCK, 1);
    dim3 blocks(divup(n_corners, threads.x), 1);

    keepCornersKernel<<<blocks, threads>>>(x_out, y_out, resp_out,
                                           x_in, y_in, resp_in, resp_idx, n_corners);

    POST_LAUNCH_CHECK();
}

}

}
