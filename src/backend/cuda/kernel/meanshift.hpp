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
#include "shared.hpp"

namespace cuda
{

namespace kernel
{

static const dim_type THREADS_X = 16;
static const dim_type THREADS_Y = 16;

__forceinline__ __device__ 
dim_type lIdx(dim_type x, dim_type y,
              dim_type stride1, dim_type stride0)
{
    return (y*stride1 + x*stride0);
}

__forceinline__ __device__ 
dim_type clamp(dim_type f, dim_type a, dim_type b)
{
    return max(a, min(f, b));
}

template<typename T, dim_type channels>
inline __device__ 
void load2ShrdMem(T * shrd, const T * in,
                  dim_type lx, dim_type ly, 
                  dim_type shrdStride, dim_type schStride,
                  dim_type dim0, dim_type dim1,
                  dim_type gx, dim_type gy,
                  dim_type ichStride, dim_type inStride1, dim_type inStride0)
{
    int gx_  = clamp(gx, 0, dim0-1);
    int gy_  = clamp(gy, 0, dim1-1);
#pragma unroll
    for(dim_type ch=0; ch<channels; ++ch)
        shrd[lIdx(lx, ly, shrdStride, 1)+ch*schStride] = in[lIdx(gx_, gy_, inStride1, inStride0)+ch*ichStride];
}

template<typename T, dim_type channels, dim_type batchIndex>
static __global__
void meanshiftKernel(Param<T> out, CParam<T> in,
                     float space_, dim_type radius, float cvar,
                     uint iter, dim_type nonBatchBlkSize)
{
    SharedMemory<T> shared;
    T * shrdMem = shared.getPointer();

    // calculate necessary offset and window parameters
    const dim_type padding     = 2*radius;
    const dim_type shrdLen     = blockDim.x + padding;
    const dim_type schStride   = shrdLen*(blockDim.y + padding);
    // the variable ichStride will only effect when we have >1
    // channels. in the other cases, the expression in question
    // will not use the variable
    const dim_type ichStride   = in.strides[batchIndex-1];

    // gfor batch offsets
    unsigned batchId = blockIdx.x / nonBatchBlkSize;
    const T* iptr    = (const T *) in.ptr + (batchId *  in.strides[batchIndex]);
    T*       optr    = (T *      )out.ptr + (batchId * out.strides[batchIndex]);

    const dim_type lx = threadIdx.x;
    const dim_type ly = threadIdx.y;

    const dim_type gx = blockDim.x * (blockIdx.x-batchId*nonBatchBlkSize) + lx;
    const dim_type gy = blockDim.y * blockIdx.y + ly;

    dim_type gx2 = gx + blockDim.x;
    dim_type gy2 = gy + blockDim.y;
    dim_type lx2 = lx + blockDim.x;
    dim_type ly2 = ly + blockDim.y;
    dim_type i   = lx + radius;
    dim_type j   = ly + radius;

    // pull image to local memory
    load2ShrdMem<T, channels>(shrdMem, iptr, lx, ly, shrdLen, schStride,
                              in.dims[0], in.dims[1], gx-radius,
                              gy-radius, ichStride, in.strides[1], in.strides[0]);
    if (lx<padding) {
        load2ShrdMem<T, channels>(shrdMem, iptr, lx2, ly, shrdLen, schStride,
                                  in.dims[0], in.dims[1], gx2-radius,
                                  gy-radius, ichStride, in.strides[1], in.strides[0]);
    }
    if (ly<padding) {
        load2ShrdMem<T, channels>(shrdMem, iptr, lx, ly2, shrdLen, schStride,
                                  in.dims[0], in.dims[1], gx-radius,
                                  gy2-radius, ichStride, in.strides[1], in.strides[0]);
    }
    if (lx<padding && ly<padding) {
        load2ShrdMem<T, channels>(shrdMem, iptr, lx2, ly2, shrdLen, schStride,
                                  in.dims[0], in.dims[1], gx2-radius,
                                  gy2-radius, ichStride, in.strides[1], in.strides[0]);
    }
    __syncthreads();

    if (gx>=in.dims[0] || gy>=in.dims[1])
        return;

    float means[channels];
    float centers[channels];
    float tmpclrs[channels];

    // clear means and centers for this pixel
#pragma unroll
    for(dim_type ch=0; ch<channels; ++ch) {
        means[ch] = 0.0f;
        centers[ch] = shrdMem[lIdx(i, j, shrdLen, 1)+ch*schStride];
    }

    // scope of meanshift iterationd begin
    for(uint it=0; it<iter; ++it) {

        int count   = 0;
        int shift_x = 0;
        int shift_y = 0;

        for(dim_type wj=-radius; wj<=radius; ++wj) {
            int hit_count = 0;

            for(dim_type wi=-radius; wi<=radius; ++wi) {

                dim_type tj = j + wj;
                dim_type ti = i + wi;

                // proceed
                float norm = 0.0f;
#pragma unroll
                for(dim_type ch=0; ch<channels; ++ch) {
                    tmpclrs[ch] = shrdMem[lIdx(ti, tj, shrdLen, 1)+ch*schStride];
                    norm += (centers[ch]-tmpclrs[ch]) * (centers[ch]-tmpclrs[ch]);
                }

                if (norm<= cvar) {
#pragma unroll
                    for(dim_type ch=0; ch<channels; ++ch)
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
        for(dim_type ch=0; ch<channels; ++ch)
            means[ch] *= fcount;

        float norm = 0.f;
#pragma unroll
        for(dim_type ch=0; ch<channels; ++ch)
            norm += ((means[ch]-centers[ch])*(means[ch]-centers[ch]));

        bool stop = ((abs(shift_y-mean_y)+abs(shift_x-mean_x)) + norm) <= 1;
        shift_x = mean_x;
        shift_y = mean_y;

#pragma unroll
        for(dim_type ch=0; ch<channels; ++ch)
            centers[ch] = means[ch];
        if (stop) { break; }
    } // scope of meanshift iterations end

#pragma unroll
    for(dim_type ch=0; ch<channels; ++ch)
        optr[lIdx(gx, gy, out.strides[1], out.strides[0])+ch*ichStride] = centers[ch];
}

template<typename T, bool is_color>
void meanshift(Param<T> out, CParam<T> in, float s_sigma, float c_sigma, uint iter)
{
    static dim3 threads(kernel::THREADS_X, kernel::THREADS_Y);

    dim_type blk_x = divup(in.dims[0], THREADS_X);
    dim_type blk_y = divup(in.dims[1], THREADS_Y);

    const dim_type bIndex   = (is_color ? 3ll : 2ll);
    const dim_type bCount   = in.dims[bIndex];
    const dim_type channels = (is_color ? in.dims[2] : 1ll);

    dim3 blocks(blk_x * bCount, blk_y);

    // clamp spatical and chromatic sigma's
    float space_     = std::min(11.5f, s_sigma);
    dim_type radius  = std::max((dim_type)(space_ * 1.5f), 1ll);
    dim_type padding = 2*radius+1;
    const float cvar = c_sigma*c_sigma;
    size_t shrd_size = channels*(threads.x + padding)*(threads.y+padding)*sizeof(T);

    if (is_color)
        (meanshiftKernel<T, 3ll, 3ll>) <<<blocks, threads, shrd_size>>>(out, in, space_, radius, cvar, iter, blk_x);
    else
        (meanshiftKernel<T, 1ll, 2ll>) <<<blocks, threads, shrd_size>>>(out, in, space_, radius, cvar, iter, blk_x);

    POST_LAUNCH_CHECK();
}

}

}
