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
#include "shared.hpp"

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

template<typename T>
__global__
void susanKernel(T* out, const T* in,
                 const unsigned idim0, const unsigned idim1,
                 const unsigned radius, const float t, const float g,
                 const unsigned edge)
{
    const int rSqrd   = radius*radius;
    const int windLen = 2*radius+1;
    const int shrdLen = BLOCK_X + windLen-1;

    SharedMemory<T> shared;
    T* shrdMem = shared.getPointer();

    const unsigned lx = threadIdx.x;
    const unsigned ly = threadIdx.y;
    const unsigned gx  = blockDim.x * blockIdx.x + lx + edge;
    const unsigned gy  = blockDim.y * blockIdx.y + ly + edge;

    const unsigned nucleusIdx = (ly+radius)*shrdLen + lx+radius;
    shrdMem[nucleusIdx] = gx<idim0 && gy<idim1 ? in[gy*idim0+gx] : 0;
    T m_0 = shrdMem[nucleusIdx];

#pragma unroll
    for (int b=ly, gy2=gy; b<shrdLen; b+=BLOCK_Y, gy2+=BLOCK_Y) {
        int j = gy2-radius;
#pragma unroll
        for (int a=lx, gx2=gx; a<shrdLen; a+=BLOCK_X, gx2+=BLOCK_X) {
            int i = gx2-radius;
            shrdMem[b*shrdLen+a] = (i<idim0 && j<idim1 ? in[j*idim0+i]: m_0);
        }
    }
    __syncthreads();

    if (gx < idim0 - edge && gy < idim1 - edge) {
        unsigned idx = gy*idim0 + gx;
        float nM  = 0.0f;
#pragma unroll
        for (int p=0; p<windLen; ++p) {
#pragma unroll
            for (int q=0; q<windLen; ++q) {
                int i = p - radius;
                int j = q - radius;
                int a = lx + radius + i;
                int b = ly + radius + j;
                if (i*i + j*j < rSqrd) {
                    float c = m_0;
                    float m = shrdMem[b * shrdLen + a];
                    float exp_pow = powf((m - c)/t, 6.0f);
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
    dim3 blocks(divup(idim0-edge*2, BLOCK_X), divup(idim1-edge*2, BLOCK_Y));
    const size_t SMEM_SIZE = (BLOCK_X+2*radius)*(BLOCK_Y+2*radius)*sizeof(T);

    CUDA_LAUNCH_SMEM((susanKernel<T>), blocks, threads, SMEM_SIZE,
            out, in, idim0, idim1, radius, t, g, edge);

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

    if (gx < idim0 - r && gy < idim1 - r) {
        const T v = resp_in[gy * idim0 + gx];

        // Find maximum neighborhood response
        T max_v;
        max_v = max_val(resp_in[(gy-1) * idim0 + gx-1], resp_in[gy * idim0 + gx-1]);
        max_v = max_val(max_v, resp_in[(gy+1) * idim0 + gx-1]);
        max_v = max_val(max_v, resp_in[(gy-1) * idim0 + gx  ]);
        max_v = max_val(max_v, resp_in[(gy+1) * idim0 + gx  ]);
        max_v = max_val(max_v, resp_in[(gy-1) * idim0 + gx+1]);
        max_v = max_val(max_v, resp_in[(gy)   * idim0 + gx+1]);
        max_v = max_val(max_v, resp_in[(gy+1) * idim0 + gx+1]);

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
    dim3 blocks(divup(idim0-edge*2, BLOCK_X), divup(idim1-edge*2, BLOCK_Y));

    unsigned* d_corners_found = memAlloc<unsigned>(1);
    CUDA_CHECK(cudaMemsetAsync(d_corners_found, 0, sizeof(unsigned),
                cuda::getStream(cuda::getActiveDeviceId())));

    CUDA_LAUNCH((nonMaxKernel<T>), blocks, threads,
            x_out, y_out, resp_out, d_corners_found, idim0, idim1, resp_in, edge, max_corners);

    POST_LAUNCH_CHECK();

    CUDA_CHECK(cudaMemcpy(count, d_corners_found, sizeof(unsigned), cudaMemcpyDeviceToHost));
    memFree(d_corners_found);
}

}

}
