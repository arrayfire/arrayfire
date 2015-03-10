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
#include <convolve.hpp>

namespace cuda
{

namespace kernel
{

static const dim_type THREADS_X = 16;
static const dim_type THREADS_Y = 16;

// below shared MAX_*_LEN's are calculated based on
// a maximum shared memory configuration of 48KB per block
// considering complex types as well
static const dim_type MAX_SCONV_FILTER_LEN = 31;

// we shall declare the maximum size required of above all three cases
// and re-use the same constant memory locations for every case
__constant__ char sFilter[2*THREADS_Y*(2*(MAX_SCONV_FILTER_LEN-1)+THREADS_X)*sizeof(double)];

template<typename T, typename accType, dim_type conv_dim, bool expand, dim_type fLen>
__global__
void convolve2_separable(Param<T> out, CParam<T> signal, dim_type nBBS)
{
    const dim_type smem_len =   (conv_dim==0 ?
                                (THREADS_X+2*(fLen-1))* THREADS_Y:
                                (THREADS_Y+2*(fLen-1))* THREADS_X);
    __shared__ T shrdMem[smem_len];

    const dim_type radius  = fLen-1;
    const dim_type padding = 2*radius;
    const dim_type s0      = signal.strides[0];
    const dim_type s1      = signal.strides[1];
    const dim_type d0      = signal.dims[0];
    const dim_type d1      = signal.dims[1];
    const dim_type shrdLen = THREADS_X + (conv_dim==0 ? padding : 0);

    unsigned batchId  = blockIdx.x/nBBS;
    T *dst            = (T *)out.ptr          + (batchId*out.strides[2]);
    const T *src      = (const T *)signal.ptr + (batchId*signal.strides[2]);
    const accType *impulse  = (const accType *)sFilter;

    dim_type lx = threadIdx.x;
    dim_type ly = threadIdx.y;
    dim_type ox = THREADS_X * (blockIdx.x-batchId*nBBS) + lx;
    dim_type oy = THREADS_Y * blockIdx.y + ly;
    dim_type gx = ox;
    dim_type gy = oy;

    // below if-else statement is based on template parameter
    if (conv_dim==0) {
        gx += (expand ? 0 : fLen>>1);
        dim_type endX = ((fLen-1)<<1) + THREADS_X;

#pragma unroll
        for(dim_type lx = threadIdx.x, glb_x = gx; lx<endX; lx += THREADS_X, glb_x += THREADS_X) {
            dim_type i = glb_x - radius;
            dim_type j = gy;
            bool is_i  = i>=0 && i<d0;
            bool is_j  = j>=0 && j<d1;
            shrdMem[ly*shrdLen+lx] = (is_i && is_j ? src[i*s0 + j*s1] : scalar<T>(0));
        }

    } else if (conv_dim==1) {
        gy += (expand ? 0 : fLen>>1);
        dim_type endY = ((fLen-1)<<1) + THREADS_Y;

#pragma unroll
        for(dim_type ly = threadIdx.y, glb_y = gy; ly<endY; ly += THREADS_Y, glb_y += THREADS_Y) {
            dim_type i = gx;
            dim_type j = glb_y - radius;
            bool is_i  = i>=0 && i<d0;
            bool is_j  = j>=0 && j<d1;
            shrdMem[ly*shrdLen+lx] = (is_i && is_j ? src[i*s0 + j*s1] : scalar<T>(0));
        }
    }
    __syncthreads();

    if (ox<out.dims[0] && oy<out.dims[1]) {
        // below conditional statement is based on template parameter
        dim_type i  = (conv_dim==0 ? lx : ly) + radius;
        accType accum = scalar<accType>(0);
#pragma unroll
        for(dim_type f=0; f<fLen; ++f) {
            accType f_val = impulse[f];
            // below conditional statement is based on template parameter
            dim_type s_idx = (conv_dim==0 ? (ly*shrdLen+(i-f)) : ((i-f)*shrdLen+lx));
            T s_val = shrdMem[s_idx];
            accum   = accum + s_val*f_val;
        }
        dst[oy*out.strides[1]+ox] = (T)accum;
    }
}

template<typename T, typename aT, dim_type cDim, bool expand, dim_type f>
void conv2Helper(dim3 blks, dim3 thrds, Param<T> out, CParam<T> sig, dim_type nBBS)
{
   (convolve2_separable<T, aT, cDim, expand, f>)<<<blks, thrds>>>(out, sig, nBBS);
}

template<typename T, typename accType, dim_type conv_dim, bool expand>
void convolve2(Param<T> out, CParam<T> signal, CParam<accType> filter)
{
    dim_type fLen = filter.dims[0] * filter.dims[1] * filter.dims[2] * filter.dims[3];
    if(fLen > kernel::MAX_SCONV_FILTER_LEN) {
        // call upon fft
        CUDA_NOT_SUPPORTED();
    }

    dim3 threads(THREADS_X, THREADS_Y);

    dim_type blk_x = divup(out.dims[0], threads.x);
    dim_type blk_y = divup(out.dims[1], threads.y);

    dim3 blocks(blk_x*signal.dims[2], blk_y);


   // FIX ME: if the filter array is strided, direct copy of symbols
   // might cause issues
   CUDA_CHECK(cudaMemcpyToSymbol(kernel::sFilter, filter.ptr, fLen*sizeof(accType), 0, cudaMemcpyDeviceToDevice));

    switch(fLen) {
        case  2: conv2Helper<T, accType, conv_dim, expand,  2>(blocks, threads, out, signal, blk_x); break;
        case  3: conv2Helper<T, accType, conv_dim, expand,  3>(blocks, threads, out, signal, blk_x); break;
        case  4: conv2Helper<T, accType, conv_dim, expand,  4>(blocks, threads, out, signal, blk_x); break;
        case  5: conv2Helper<T, accType, conv_dim, expand,  5>(blocks, threads, out, signal, blk_x); break;
        case  6: conv2Helper<T, accType, conv_dim, expand,  6>(blocks, threads, out, signal, blk_x); break;
        case  7: conv2Helper<T, accType, conv_dim, expand,  7>(blocks, threads, out, signal, blk_x); break;
        case  8: conv2Helper<T, accType, conv_dim, expand,  8>(blocks, threads, out, signal, blk_x); break;
        case  9: conv2Helper<T, accType, conv_dim, expand,  9>(blocks, threads, out, signal, blk_x); break;
        case 10: conv2Helper<T, accType, conv_dim, expand, 10>(blocks, threads, out, signal, blk_x); break;
        case 11: conv2Helper<T, accType, conv_dim, expand, 11>(blocks, threads, out, signal, blk_x); break;
        case 12: conv2Helper<T, accType, conv_dim, expand, 12>(blocks, threads, out, signal, blk_x); break;
        case 13: conv2Helper<T, accType, conv_dim, expand, 13>(blocks, threads, out, signal, blk_x); break;
        case 14: conv2Helper<T, accType, conv_dim, expand, 14>(blocks, threads, out, signal, blk_x); break;
        case 15: conv2Helper<T, accType, conv_dim, expand, 15>(blocks, threads, out, signal, blk_x); break;
        case 16: conv2Helper<T, accType, conv_dim, expand, 16>(blocks, threads, out, signal, blk_x); break;
        case 17: conv2Helper<T, accType, conv_dim, expand, 17>(blocks, threads, out, signal, blk_x); break;
        case 18: conv2Helper<T, accType, conv_dim, expand, 18>(blocks, threads, out, signal, blk_x); break;
        case 19: conv2Helper<T, accType, conv_dim, expand, 19>(blocks, threads, out, signal, blk_x); break;
        case 20: conv2Helper<T, accType, conv_dim, expand, 20>(blocks, threads, out, signal, blk_x); break;
        case 21: conv2Helper<T, accType, conv_dim, expand, 21>(blocks, threads, out, signal, blk_x); break;
        case 22: conv2Helper<T, accType, conv_dim, expand, 22>(blocks, threads, out, signal, blk_x); break;
        case 23: conv2Helper<T, accType, conv_dim, expand, 23>(blocks, threads, out, signal, blk_x); break;
        case 24: conv2Helper<T, accType, conv_dim, expand, 24>(blocks, threads, out, signal, blk_x); break;
        case 25: conv2Helper<T, accType, conv_dim, expand, 25>(blocks, threads, out, signal, blk_x); break;
        case 26: conv2Helper<T, accType, conv_dim, expand, 26>(blocks, threads, out, signal, blk_x); break;
        case 27: conv2Helper<T, accType, conv_dim, expand, 27>(blocks, threads, out, signal, blk_x); break;
        case 28: conv2Helper<T, accType, conv_dim, expand, 28>(blocks, threads, out, signal, blk_x); break;
        case 29: conv2Helper<T, accType, conv_dim, expand, 29>(blocks, threads, out, signal, blk_x); break;
        case 30: conv2Helper<T, accType, conv_dim, expand, 30>(blocks, threads, out, signal, blk_x); break;
        case 31: conv2Helper<T, accType, conv_dim, expand, 31>(blocks, threads, out, signal, blk_x); break;
        default: CUDA_NOT_SUPPORTED();
    }

   POST_LAUNCH_CHECK();
}

#define INSTANTIATE(T, accType)                                         \
	template void convolve2<T, accType, 0, true >(Param<T> out, CParam<T> signal, CParam<accType> filter); \
	template void convolve2<T, accType, 0, false>(Param<T> out, CParam<T> signal, CParam<accType> filter); \
	template void convolve2<T, accType, 1, true >(Param<T> out, CParam<T> signal, CParam<accType> filter); \
	template void convolve2<T, accType, 1, false>(Param<T> out, CParam<T> signal, CParam<accType> filter); \


INSTANTIATE(cdouble, cdouble)
INSTANTIATE(cfloat ,  cfloat)
INSTANTIATE(double ,  double)
INSTANTIATE(float  ,   float)
INSTANTIATE(uint   ,   float)
INSTANTIATE(int    ,   float)
INSTANTIATE(uchar  ,   float)
INSTANTIATE(char   ,   float)

}

}
