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

inline __device__
dim_type lIdx(dim_type x, dim_type y, dim_type stride1, dim_type stride0)
{
    return (y*stride1 + x*stride0);
}

inline __device__
dim_type clamp(dim_type f, dim_type a, dim_type b)
{
    return max(a, min(f, b));
}

inline __device__
float gaussian1d(float x, float variance)
{
    return exp((x * x) / (-2.f * variance));
}

template<typename inType, typename outType>
inline __device__
void load2ShrdMem(outType * shrd, const inType * const in,
                  dim_type lx, dim_type ly, dim_type shrdStride,
                  dim_type dim0, dim_type dim1,
                  dim_type gx, dim_type gy,
                  dim_type inStride1, dim_type inStride0)
{
    shrd[ly*shrdStride+lx] = in[lIdx(clamp(gx, 0, dim0-1), clamp(gy, 0, dim1-1), inStride1, inStride0)];
}

template<typename inType, typename outType>
static __global__
void bilateralKernel(Param<outType> out, CParam<inType> in,
                     float sigma_space, float sigma_color,
                     dim_type gaussOff, dim_type nonBatchBlkSize)
{
    SharedMemory<outType> shared;
    outType *localMem = shared.getPointer();
    outType *gauss2d  = localMem + gaussOff;

    const dim_type radius      = max((int)(sigma_space * 1.5f), 1);
    const dim_type padding     = 2 * radius;
    const dim_type window_size = padding + 1;
    const dim_type shrdLen     = THREADS_X + padding;
    const float variance_range = sigma_color * sigma_color;
    const float variance_space = sigma_space * sigma_space;

    // gfor batch offsets
    unsigned batchId = blockIdx.x / nonBatchBlkSize;
    const inType* iptr  = (const inType *) in.ptr + (batchId * in.strides[2]);
    outType*       optr = (outType *     )out.ptr + (batchId * out.strides[2]);

    dim_type lx = threadIdx.x;
    dim_type ly = threadIdx.y;

    const dim_type gx = THREADS_X * (blockIdx.x-batchId*nonBatchBlkSize) + lx;
    const dim_type gy = THREADS_Y * blockIdx.y + ly;

    // generate gauss2d spatial variance values for block
    if (lx<window_size && ly<window_size) {
        int x = lx - radius;
        int y = ly - radius;
        gauss2d[ly*window_size+lx] = exp( ((x*x) + (y*y)) / (-2.f * variance_space));
    }

    // pull image to local memory
    load2ShrdMem<inType, outType>(localMem, iptr, lx, ly, shrdLen,
            in.dims[0], in.dims[1], gx-radius, gy-radius, in.strides[1], in.strides[0]);

    dim_type lx2 = lx + THREADS_X;
    dim_type ly2 = ly + THREADS_Y;
    dim_type gx2 = gx + THREADS_X;
    dim_type gy2 = gy + THREADS_Y;

    if (lx<padding) {
        load2ShrdMem<inType, outType>(localMem, iptr, lx2, ly, shrdLen,
                in.dims[0], in.dims[1], gx2-radius, gy-radius, in.strides[1], in.strides[0]);
    }
    if (ly<padding) {
        load2ShrdMem<inType, outType>(localMem, iptr, lx, ly2, shrdLen,
                in.dims[0], in.dims[1], gx-radius, gy2-radius, in.strides[1], in.strides[0]);
    }
    if (lx<padding && ly<padding) {
        load2ShrdMem<inType, outType>(localMem, iptr, lx2, ly2, shrdLen,
                in.dims[0], in.dims[1], gx2-radius, gy2-radius, in.strides[1], in.strides[0]);
    }
    __syncthreads();

    if (gx<in.dims[0] && gy<in.dims[1]) {
        lx += radius;
        ly += radius;
        const outType center_color = localMem[ly*shrdLen+lx];
        outType res  = 0;
        outType norm = 0;
#pragma unroll
        for(dim_type wj=0; wj<window_size; ++wj) {
            dim_type joff = (ly+wj-radius)*shrdLen + (lx-radius);
            dim_type goff = wj*window_size;
#pragma unroll
            for(dim_type wi=0; wi<window_size; ++wi) {
                const outType tmp_color   = localMem[joff+wi];
                const outType gauss_range = gaussian1d(center_color - tmp_color, variance_range);
                const outType weight      = gauss2d[goff+wi] * gauss_range;
                norm += weight;
                res  += tmp_color * weight;
            }
        }
        optr[gy*out.strides[1]+gx] = res / norm;
    }
}

template<typename inType, typename outType, bool isColor>
void bilateral(Param<outType> out, CParam<inType> in, float s_sigma, float c_sigma)
{
    dim3 threads(kernel::THREADS_X, kernel::THREADS_Y);

    dim_type blk_x = divup(in.dims[0], THREADS_X);
    dim_type blk_y = divup(in.dims[1], THREADS_Y);
    dim_type bCount= blk_x * (isColor ? in.dims[3] : in.dims[2]*in.dims[3]);

    dim3 blocks(bCount, blk_y);

    // calculate shared memory size
    dim_type radius = (dim_type)std::max(s_sigma * 1.5f, 1.f);
    dim_type num_shrd_elems    = (THREADS_X + 2 * radius) * (THREADS_Y + 2 * radius);
    dim_type num_gauss_elems   = (2 * radius + 1)*(2 * radius + 1);
    dim_type total_shrd_size   = sizeof(outType) * (num_shrd_elems + num_gauss_elems);

    bilateralKernel<inType, outType>
        <<<blocks, threads, total_shrd_size>>>
        (out, in, s_sigma, c_sigma, num_shrd_elems, blk_x);

    POST_LAUNCH_CHECK();
}

}

}
