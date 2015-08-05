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
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_cuda.hpp>
#include <math.hpp>
#include "shared.hpp"

namespace cuda
{

namespace kernel
{

static const int THREADS_X = 16;
static const int THREADS_Y = 16;

__forceinline__ __device__
int lIdx(int x, int y,
              int stride1, int stride0)
{
    return (y*stride1 + x*stride0);
}

__forceinline__ __device__
int clamp(int f, int a, int b)
{
    return max(a, min(f, b));
}

template<typename T, int channels>
inline __device__
void load2ShrdMem(T * shrd, const T * in,
                  int lx, int ly,
                  int shrdStride, int schStride,
                  int dim0, int dim1,
                  int gx, int gy,
                  int ichStride, int inStride1, int inStride0)
{
    int gx_  = clamp(gx, 0, dim0-1);
    int gy_  = clamp(gy, 0, dim1-1);
#pragma unroll
    for(int ch=0; ch<channels; ++ch)
        shrd[lIdx(lx, ly, shrdStride, 1)+ch*schStride] = in[lIdx(gx_, gy_, inStride1, inStride0)+ch*ichStride];
}

template<typename T, int channels>
static __global__
void meanshiftKernel(Param<T> out, CParam<T> in,
                     float space_, int radius, float cvar,
                     uint iter, int nBBS0, int nBBS1)
{
    SharedMemory<T> shared;
    T * shrdMem = shared.getPointer();

    // calculate necessary offset and window parameters
    const int padding     = 2*radius + 1;
    const int shrdLen     = blockDim.x + padding;
    const int schStride   = shrdLen*(blockDim.y + padding);
    // the variable ichStride will only effect when we have >1
    // channels. in the other cases, the expression in question
    // will not use the variable
    const int ichStride   = in.strides[2];

    // gfor batch offsets
    unsigned b2 = blockIdx.x / nBBS0;
    unsigned b3 = blockIdx.y / nBBS1;
    const T* iptr = (const T *) in.ptr + (b2 *  in.strides[2] + b3 *  in.strides[3]);
    T*       optr = (T *      )out.ptr + (b2 * out.strides[2] + b3 * out.strides[3]);

    const int lx = threadIdx.x;
    const int ly = threadIdx.y;

    const int gx = blockDim.x * (blockIdx.x-b2*nBBS0) + lx;
    const int gy = blockDim.y * (blockIdx.y-b3*nBBS1) + ly;

    // pull image to local memory
    for (int b=ly, gy2=gy; b<shrdLen; b+=blockDim.y, gy2+=blockDim.y) {
        // move row_set get_local_size(1) along coloumns
        for (int a=lx, gx2=gx; a<shrdLen; a+=blockDim.x, gx2+=blockDim.x) {
            load2ShrdMem<T, channels>(shrdMem, iptr, a, b, shrdLen, schStride,
                    in.dims[0], in.dims[1], gx2-radius, gy2-radius, ichStride,
                    in.strides[1], in.strides[0]);
        }
    }

    int i   = lx + radius;
    int j   = ly + radius;

    __syncthreads();

    if (gx>=in.dims[0] || gy>=in.dims[1])
        return;

    float means[channels];
    float centers[channels];
    float tmpclrs[channels];

    // clear means and centers for this pixel
#pragma unroll
    for(int ch=0; ch<channels; ++ch) {
        means[ch] = 0.0f;
        centers[ch] = shrdMem[lIdx(i, j, shrdLen, 1)+ch*schStride];
    }

    // scope of meanshift iterationd begin
    for(uint it=0; it<iter; ++it) {

        int count   = 0;
        int shift_x = 0;
        int shift_y = 0;

        for(int wj=-radius; wj<=radius; ++wj) {
            int hit_count = 0;

            for(int wi=-radius; wi<=radius; ++wi) {

                int tj = j + wj;
                int ti = i + wi;

                // proceed
                float norm = 0.0f;
#pragma unroll
                for(int ch=0; ch<channels; ++ch) {
                    tmpclrs[ch] = shrdMem[lIdx(ti, tj, shrdLen, 1)+ch*schStride];
                    norm += (centers[ch]-tmpclrs[ch]) * (centers[ch]-tmpclrs[ch]);
                }

                if (norm<= cvar) {
#pragma unroll
                    for(int ch=0; ch<channels; ++ch)
                        means[ch] += tmpclrs[ch];

                    shift_x += wi;
                    ++hit_count;
                }
            }
            count+= hit_count;
            shift_y += wj*hit_count;
        }

        if (count==0) { break; }

        const float fcount = 1.f/count;
        const int mean_x = (int)(shift_x*fcount+0.5f);
        const int mean_y = (int)(shift_y*fcount+0.5f);
#pragma unroll
        for(int ch=0; ch<channels; ++ch)
            means[ch] *= fcount;

        float norm = 0.f;
#pragma unroll
        for(int ch=0; ch<channels; ++ch)
            norm += ((means[ch]-centers[ch])*(means[ch]-centers[ch]));

        bool stop = ((abs(shift_y-mean_y)+abs(shift_x-mean_x)) + norm) <= 1;
        shift_x = mean_x;
        shift_y = mean_y;

#pragma unroll
        for(int ch=0; ch<channels; ++ch)
            centers[ch] = means[ch];
        if (stop) { break; }
    } // scope of meanshift iterations end

#pragma unroll
    for(int ch=0; ch<channels; ++ch)
        optr[lIdx(gx, gy, out.strides[1], out.strides[0])+ch*ichStride] = centers[ch];
}

template<typename T, bool is_color>
void meanshift(Param<T> out, CParam<T> in, float s_sigma, float c_sigma, uint iter)
{
    static dim3 threads(kernel::THREADS_X, kernel::THREADS_Y);

    int blk_x = divup(in.dims[0], THREADS_X);
    int blk_y = divup(in.dims[1], THREADS_Y);

    const int bCount   = (is_color ? 1 : in.dims[2]);
    const int channels = (is_color ? in.dims[2] : 1); // this has to be 3 for color images

    dim3 blocks(blk_x * bCount, blk_y * in.dims[3]);

    // clamp spatical and chromatic sigma's
    float space_     = std::min(11.5f, s_sigma);
    int radius  = std::max((int)(space_ * 1.5f), 1);
    int padding = 2*radius+1;
    const float cvar = c_sigma*c_sigma;
    size_t shrd_size = channels*(threads.x + padding)*(threads.y+padding)*sizeof(T);

    if (is_color)
        CUDA_LAUNCH_SMEM((meanshiftKernel<T, 3>), blocks, threads, shrd_size,
                out, in, space_, radius, cvar, iter, blk_x, blk_y);
    else
        CUDA_LAUNCH_SMEM((meanshiftKernel<T, 1>), blocks, threads, shrd_size,
                out, in, space_, radius, cvar, iter, blk_x, blk_y);

    POST_LAUNCH_CHECK();
}

}

}
