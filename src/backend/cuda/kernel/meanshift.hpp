/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <backend.hpp>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_cuda.hpp>
#include <math.hpp>

#include <type_traits>

namespace cuda
{
namespace kernel
{
static const int THREADS_X = 32;
static const int THREADS_Y = 8;

template<typename AccType, typename T, int channels>
static __global__
void meanshiftKernel(Param<T> out, CParam<T> in, int radius, float cvar, uint iter,
                     int nBBS0, int nBBS1)
{
    unsigned b2   = blockIdx.x / nBBS0;
    unsigned b3   = blockIdx.y / nBBS1;
    const T* iptr = (const T *) in.ptr + (b2 *  in.strides[2] + b3 *  in.strides[3]);
    T*       optr = (T *      )out.ptr + (b2 * out.strides[2] + b3 * out.strides[3]);
    const int gx  = blockDim.x * (blockIdx.x-b2*nBBS0) + threadIdx.x;
    const int gy  = blockDim.y * (blockIdx.y-b3*nBBS1) + threadIdx.y;

    if (gx>=in.dims[0] || gy>=in.dims[1])
        return;

    int i = gx;
    int j = gy;

    T centers[channels];
    T tmpclrs[channels];

    AccType means[channels];

#pragma unroll
    for (int ch=0; ch<channels; ++ch)
        centers[ch] = iptr[ (gx*in.strides[0] + gy*in.strides[1] + ch*in.strides[2]) ];

    const int dim0LenLmt = in.dims[0]-1;
    const int dim1LenLmt = in.dims[1]-1;

    // scope of meanshift iterations begin
    for (uint it=0; it<iter; ++it) {

        int ocj = j;
        int oci = i;
        unsigned count  = 0;

        int shift_x = 0;
        int shift_y = 0;

#pragma unroll
        for (int ch=0; ch<channels; ++ch)
            means[ch] = 0;

        for (int wj=-radius; wj<=radius; ++wj) {
            int hit_count = 0;
            int tj = j + wj;

            if (tj<0 || tj>dim1LenLmt) break;

            for(int wi=-radius; wi<=radius; ++wi) {

                int ti = i + wi;

                if (ti<0 || ti>dim0LenLmt) break;

                AccType norm = 0;
#pragma unroll
                for (int ch=0; ch<channels; ++ch) {
                    tmpclrs[ch] = iptr[ (ti*in.strides[0] + tj*in.strides[1] + ch*in.strides[2]) ];
                    AccType diff = (AccType)centers[ch] - (AccType)tmpclrs[ch];
                    norm += (diff * diff);
                }

                if (norm <= cvar) {
#pragma unroll
                    for (int ch=0; ch<channels; ++ch)
                        means[ch] += (AccType)tmpclrs[ch];

                    shift_x += ti;
                    ++hit_count;
                }
            }
            count += hit_count;
            shift_y += tj*hit_count;
        }

        if (count==0) break;

        const AccType fcount = 1/(AccType)count;

        i = __float2int_rz(shift_x*fcount);
        j = __float2int_rz(shift_y*fcount);

#pragma unroll
        for (int ch=0; ch<channels; ++ch)
            means[ch] = __float2int_rz(means[ch]*fcount);

        AccType norm = 0;
#pragma unroll
        for (int ch=0; ch<channels; ++ch) {
            AccType diff = (AccType)centers[ch] - means[ch];
            norm += (diff * diff);
        }

        bool stop = (j==ocj && i==oci) || ((abs(ocj-j) + abs(oci-i) + norm) <= 1);

#pragma unroll
        for (int ch=0; ch<channels; ++ch)
            centers[ch] = (T)(means[ch]);

        if (stop) break;
    } // scope of meanshift iterations end

#pragma unroll
    for (int ch=0; ch<channels; ++ch)
        optr[ (gx*out.strides[0] + gy*out.strides[1] + ch*out.strides[2]) ] = centers[ch];
}

template<typename T, bool is_color>
void meanshift(Param<T> out, CParam<T> in, float s_sigma, float c_sigma, uint iter)
{
    typedef typename std::conditional< std::is_same<T, double>::value, double, float >::type AccType;

    static dim3 threads(kernel::THREADS_X, kernel::THREADS_Y);

    int blk_x = divup(in.dims[0], THREADS_X);
    int blk_y = divup(in.dims[1], THREADS_Y);

    const int bCount   = (is_color ? 1 : in.dims[2]);

    dim3 blocks(blk_x * bCount, blk_y * in.dims[3]);

    // clamp spatical and chromatic sigma's
    int radius   = std::max( (int)(std::min(11.5f, s_sigma) * 1.5f), 1 );

    const float cvar = c_sigma*c_sigma;

    if (is_color)
        CUDA_LAUNCH((meanshiftKernel<AccType, T, 3>), blocks, threads,
                out, in, radius, cvar, iter, blk_x, blk_y);
    else
        CUDA_LAUNCH((meanshiftKernel<AccType, T, 1>), blocks, threads,
                out, in, radius, cvar, iter, blk_x, blk_y);

    POST_LAUNCH_CHECK();
}
}
}
