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
static const dim_type THREADS_X =  256;
static const dim_type THRD_LOAD =   16;

__forceinline__ __device__ dim_type minimum(dim_type a, dim_type b)
{
  return (a < b ? a : b);
}

template<typename inType, typename outType>
static __global__
void histogramKernel(Param<outType> out, CParam<inType> in,
                     const cfloat *d_minmax, dim_type len,
                     dim_type nbins, dim_type blk_x)
{
    SharedMemory<outType> shared;
    outType * shrdMem = shared.getPointer();

    // offset minmax array to account for batch ops
    d_minmax += blockIdx.y;

    // offset input and output to account for batch ops
    const inType *iptr  =  in.ptr + blockIdx.y *  in.strides[2];
    outType      *optr  = out.ptr + blockIdx.y * out.strides[2];

    int start = blockIdx.x * THRD_LOAD * blockDim.x + threadIdx.x;
    int end   = minimum((start + THRD_LOAD * blockDim.x), len);

    __shared__ float min;
    __shared__ float step;

    if (threadIdx.x == 0) {
        float2 minmax = *d_minmax;
        min  = minmax.x;
        step = (minmax.y-minmax.x) / (float)nbins;
    }

    for (int i = threadIdx.x; i < nbins; i += blockDim.x)
        shrdMem[i] = 0;
    __syncthreads();

    for (int row = start; row < end; row += blockDim.x) {
        int bin = (int)((iptr[row] - min) / step);
        bin     = (bin < 0)      ? 0         : bin;
        bin     = (bin >= nbins) ? (nbins-1) : bin;
        atomicAdd((shrdMem + bin), 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < nbins; i += blockDim.x) {
        atomicAdd((optr + i), shrdMem[i]);
    }
}

template<typename inType, typename outType>
void histogram(Param<outType> out, CParam<inType> in, cfloat *d_minmax, dim_type nbins)
{
    dim3 threads(kernel::THREADS_X, 1);
    dim_type numElements= in.dims[0] * in.dims[1];
    dim_type blk_x = divup(numElements, THRD_LOAD*THREADS_X);
    dim_type batchCount = in.dims[2];
    dim3 blocks(blk_x, batchCount);
    dim_type smem_size = nbins * sizeof(outType);

    histogramKernel<inType, outType>
        <<<blocks, threads, smem_size>>>
        (out, in, d_minmax, numElements, nbins, blk_x);

    POST_LAUNCH_CHECK();
}

}

}
