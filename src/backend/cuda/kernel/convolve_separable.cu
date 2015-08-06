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

static const int THREADS_X = 16;
static const int THREADS_Y = 16;

// below shared MAX_*_LEN's are calculated based on
// a maximum shared memory configuration of 48KB per block
// considering complex types as well
static const int MAX_SCONV_FILTER_LEN = 31;

// we shall declare the maximum size required of above all three cases
// and re-use the same constant memory locations for every case
__constant__ char sFilter[2*THREADS_Y*(2*(MAX_SCONV_FILTER_LEN-1)+THREADS_X)*sizeof(double)];

template<typename T, typename accType, int conv_dim, bool expand, int fLen>
__global__
void convolve2_separable(Param<T> out, CParam<T> signal, int nBBS0, int nBBS1)
{
    const int smem_len =   (conv_dim==0 ?
                                (THREADS_X+2*(fLen-1))* THREADS_Y:
                                (THREADS_Y+2*(fLen-1))* THREADS_X);
    __shared__ T shrdMem[smem_len];

    const int radius  = fLen-1;
    const int padding = 2*radius;
    const int s0      = signal.strides[0];
    const int s1      = signal.strides[1];
    const int d0      = signal.dims[0];
    const int d1      = signal.dims[1];
    const int shrdLen = THREADS_X + (conv_dim==0 ? padding : 0);

    unsigned b2  = blockIdx.x/nBBS0;
    unsigned b3  = blockIdx.y/nBBS1;
    T *dst       = (T *)out.ptr          + (b2*out.strides[2] + b3*out.strides[3]);
    const T *src = (const T *)signal.ptr + (b2*signal.strides[2] + b3*signal.strides[3]);
    const accType *impulse  = (const accType *)sFilter;

    int lx = threadIdx.x;
    int ly = threadIdx.y;
    int ox = THREADS_X * (blockIdx.x-b2*nBBS0) + lx;
    int oy = THREADS_Y * (blockIdx.y-b3*nBBS1) + ly;
    int gx = ox;
    int gy = oy;

    // below if-else statement is based on template parameter
    if (conv_dim==0) {
        gx += (expand ? 0 : fLen>>1);
        int endX = ((fLen-1)<<1) + THREADS_X;

#pragma unroll
        for(int lx = threadIdx.x, glb_x = gx; lx<endX; lx += THREADS_X, glb_x += THREADS_X) {
            int i = glb_x - radius;
            int j = gy;
            bool is_i  = i>=0 && i<d0;
            bool is_j  = j>=0 && j<d1;
            shrdMem[ly*shrdLen+lx] = (is_i && is_j ? src[i*s0 + j*s1] : scalar<T>(0));
        }

    } else if (conv_dim==1) {
        gy += (expand ? 0 : fLen>>1);
        int endY = ((fLen-1)<<1) + THREADS_Y;

#pragma unroll
        for(int ly = threadIdx.y, glb_y = gy; ly<endY; ly += THREADS_Y, glb_y += THREADS_Y) {
            int i = gx;
            int j = glb_y - radius;
            bool is_i  = i>=0 && i<d0;
            bool is_j  = j>=0 && j<d1;
            shrdMem[ly*shrdLen+lx] = (is_i && is_j ? src[i*s0 + j*s1] : scalar<T>(0));
        }
    }
    __syncthreads();

    if (ox<out.dims[0] && oy<out.dims[1]) {
        // below conditional statement is based on template parameter
        int i  = (conv_dim==0 ? lx : ly) + radius;
        accType accum = scalar<accType>(0);
#pragma unroll
        for(int f=0; f<fLen; ++f) {
            accType f_val = impulse[f];
            // below conditional statement is based on template parameter
            int s_idx = (conv_dim==0 ? (ly*shrdLen+(i-f)) : ((i-f)*shrdLen+lx));
            T s_val = shrdMem[s_idx];
            accum   = accum + s_val*f_val;
        }
        dst[oy*out.strides[1]+ox] = (T)accum;
    }
}

template<typename T, typename aT, int cDim, bool expand, int f>
void conv2Helper(dim3 blks, dim3 thrds, Param<T> out, CParam<T> sig, int nBBS0, int nBBS1)
{
   CUDA_LAUNCH((convolve2_separable<T, aT, cDim, expand, f>), blks, thrds, out, sig, nBBS0, nBBS1);
}

template<typename T, typename accType, int conv_dim, bool expand>
void convolve2(Param<T> out, CParam<T> signal, CParam<accType> filter)
{
    int fLen = filter.dims[0] * filter.dims[1] * filter.dims[2] * filter.dims[3];
    if(fLen > kernel::MAX_SCONV_FILTER_LEN) {
        // call upon fft
        CUDA_NOT_SUPPORTED();
    }

    dim3 threads(THREADS_X, THREADS_Y);

    int blk_x = divup(out.dims[0], threads.x);
    int blk_y = divup(out.dims[1], threads.y);

    dim3 blocks(blk_x*signal.dims[2], blk_y*signal.dims[3]);


   // FIX ME: if the filter array is strided, direct copy of symbols
   // might cause issues
   CUDA_CHECK(cudaMemcpyToSymbolAsync(kernel::sFilter, filter.ptr, fLen*sizeof(accType), 0,
               cudaMemcpyDeviceToDevice, cuda::getStream(cuda::getActiveDeviceId())));

    switch(fLen) {
        case  2: conv2Helper<T, accType, conv_dim, expand,  2>(blocks, threads, out, signal, blk_x, blk_y); break;
        case  3: conv2Helper<T, accType, conv_dim, expand,  3>(blocks, threads, out, signal, blk_x, blk_y); break;
        case  4: conv2Helper<T, accType, conv_dim, expand,  4>(blocks, threads, out, signal, blk_x, blk_y); break;
        case  5: conv2Helper<T, accType, conv_dim, expand,  5>(blocks, threads, out, signal, blk_x, blk_y); break;
        case  6: conv2Helper<T, accType, conv_dim, expand,  6>(blocks, threads, out, signal, blk_x, blk_y); break;
        case  7: conv2Helper<T, accType, conv_dim, expand,  7>(blocks, threads, out, signal, blk_x, blk_y); break;
        case  8: conv2Helper<T, accType, conv_dim, expand,  8>(blocks, threads, out, signal, blk_x, blk_y); break;
        case  9: conv2Helper<T, accType, conv_dim, expand,  9>(blocks, threads, out, signal, blk_x, blk_y); break;
        case 10: conv2Helper<T, accType, conv_dim, expand, 10>(blocks, threads, out, signal, blk_x, blk_y); break;
        case 11: conv2Helper<T, accType, conv_dim, expand, 11>(blocks, threads, out, signal, blk_x, blk_y); break;
        case 12: conv2Helper<T, accType, conv_dim, expand, 12>(blocks, threads, out, signal, blk_x, blk_y); break;
        case 13: conv2Helper<T, accType, conv_dim, expand, 13>(blocks, threads, out, signal, blk_x, blk_y); break;
        case 14: conv2Helper<T, accType, conv_dim, expand, 14>(blocks, threads, out, signal, blk_x, blk_y); break;
        case 15: conv2Helper<T, accType, conv_dim, expand, 15>(blocks, threads, out, signal, blk_x, blk_y); break;
        case 16: conv2Helper<T, accType, conv_dim, expand, 16>(blocks, threads, out, signal, blk_x, blk_y); break;
        case 17: conv2Helper<T, accType, conv_dim, expand, 17>(blocks, threads, out, signal, blk_x, blk_y); break;
        case 18: conv2Helper<T, accType, conv_dim, expand, 18>(blocks, threads, out, signal, blk_x, blk_y); break;
        case 19: conv2Helper<T, accType, conv_dim, expand, 19>(blocks, threads, out, signal, blk_x, blk_y); break;
        case 20: conv2Helper<T, accType, conv_dim, expand, 20>(blocks, threads, out, signal, blk_x, blk_y); break;
        case 21: conv2Helper<T, accType, conv_dim, expand, 21>(blocks, threads, out, signal, blk_x, blk_y); break;
        case 22: conv2Helper<T, accType, conv_dim, expand, 22>(blocks, threads, out, signal, blk_x, blk_y); break;
        case 23: conv2Helper<T, accType, conv_dim, expand, 23>(blocks, threads, out, signal, blk_x, blk_y); break;
        case 24: conv2Helper<T, accType, conv_dim, expand, 24>(blocks, threads, out, signal, blk_x, blk_y); break;
        case 25: conv2Helper<T, accType, conv_dim, expand, 25>(blocks, threads, out, signal, blk_x, blk_y); break;
        case 26: conv2Helper<T, accType, conv_dim, expand, 26>(blocks, threads, out, signal, blk_x, blk_y); break;
        case 27: conv2Helper<T, accType, conv_dim, expand, 27>(blocks, threads, out, signal, blk_x, blk_y); break;
        case 28: conv2Helper<T, accType, conv_dim, expand, 28>(blocks, threads, out, signal, blk_x, blk_y); break;
        case 29: conv2Helper<T, accType, conv_dim, expand, 29>(blocks, threads, out, signal, blk_x, blk_y); break;
        case 30: conv2Helper<T, accType, conv_dim, expand, 30>(blocks, threads, out, signal, blk_x, blk_y); break;
        case 31: conv2Helper<T, accType, conv_dim, expand, 31>(blocks, threads, out, signal, blk_x, blk_y); break;
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
