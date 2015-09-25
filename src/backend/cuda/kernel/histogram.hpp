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
#include <debug_cuda.hpp>
#include "shared.hpp"

namespace cuda
{

namespace kernel
{

static const unsigned MAX_BINS  = 4000;
static const int THREADS_X =  256;
static const int THRD_LOAD =   16;

__forceinline__ __device__ int minimum(int a, int b)
{
  return (a < b ? a : b);
}

template<typename inType, typename outType, bool isLinear>
static __global__
void histogramKernel(Param<outType> out, CParam<inType> in,
                     int len, int nbins, float minval, float maxval, int nBBS)
{
    SharedMemory<outType> shared;
    outType * shrdMem = shared.getPointer();

    // offset input and output to account for batch ops
    unsigned b2 = blockIdx.x / nBBS;
    const inType *iptr  =  in.ptr + b2 *  in.strides[2] + blockIdx.y *  in.strides[3];
    outType      *optr  = out.ptr + b2 * out.strides[2] + blockIdx.y * out.strides[3];

    int start  = (blockIdx.x-b2*nBBS) * THRD_LOAD * blockDim.x + threadIdx.x;
    int end    = minimum((start + THRD_LOAD * blockDim.x), len);
    float step = (maxval-minval) / (float)nbins;

    for (int i = threadIdx.x; i < nbins; i += blockDim.x)
        shrdMem[i] = 0;
    __syncthreads();

    for (int row = start; row < end; row += blockDim.x) {
        int idx = isLinear ? row : ((row % in.dims[0]) + (row / in.dims[0])*in.strides[1]);
        int bin = (int)((iptr[idx] - minval) / step);
        bin     = (bin < 0)      ? 0         : bin;
        bin     = (bin >= nbins) ? (nbins-1) : bin;
        atomicAdd((shrdMem + bin), 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < nbins; i += blockDim.x) {
        atomicAdd((optr + i), shrdMem[i]);
    }
}

template<typename inType, typename outType, bool isLinear>
void histogram(Param<outType> out, CParam<inType> in, int nbins, float minval, float maxval)
{
    dim3 threads(kernel::THREADS_X, 1);

    int nElems = in.dims[0] * in.dims[1];
    int blk_x  = divup(nElems, THRD_LOAD*THREADS_X);

    dim3 blocks(blk_x * in.dims[2], in.dims[3]);

    int smem_size = nbins * sizeof(outType);

    CUDA_LAUNCH_SMEM((histogramKernel<inType, outType, isLinear>), blocks, threads, smem_size,
            out, in, nElems, nbins, minval, maxval, blk_x);

    POST_LAUNCH_CHECK();
}

}

}
