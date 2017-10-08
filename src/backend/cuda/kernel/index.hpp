/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <backend.hpp>
#include <common/dispatch.hpp>
#include <Param.hpp>
#include <math.hpp>
#include <utility.hpp>
#include <debug_cuda.hpp>

namespace cuda
{

namespace kernel
{

static const int THREADS_X = 32;
static const int THREADS_Y =  8;

typedef struct {
    int  offs[4];
    int strds[4];
    bool     isSeq[4];
    uint*      ptr[4];
} IndexKernelParam_t;

template<typename T>
__global__
void indexKernel(Param<T> out, CParam<T> in, const IndexKernelParam_t p,
                 const int nBBS0, const int nBBS1)
{
    // retrieve index pointers
    // these can be 0 where af_array index is not used
    const uint* ptr0 = p.ptr[0];
    const uint* ptr1 = p.ptr[1];
    const uint* ptr2 = p.ptr[2];
    const uint* ptr3 = p.ptr[3];
    // retrive booleans that tell us which index to use
    const bool s0 = p.isSeq[0];
    const bool s1 = p.isSeq[1];
    const bool s2 = p.isSeq[2];
    const bool s3 = p.isSeq[3];

    const int gz = blockIdx.x/nBBS0;
    const int gx = blockDim.x * (blockIdx.x - gz*nBBS0) + threadIdx.x;

    const int gw = (blockIdx.y + blockIdx.z * gridDim.y) /nBBS1;
    const int gy = blockDim.y * ((blockIdx.y + blockIdx.z * gridDim.y) - gw*nBBS1) + threadIdx.y;

    if (gx<out.dims[0] && gy<out.dims[1] && gz<out.dims[2] && gw<out.dims[3]) {
        // calculate pointer offsets for input
        int i = p.strds[0] * trimIndex(s0 ? gx+p.offs[0] : ptr0[gx], in.dims[0]);
        int j = p.strds[1] * trimIndex(s1 ? gy+p.offs[1] : ptr1[gy], in.dims[1]);
        int k = p.strds[2] * trimIndex(s2 ? gz+p.offs[2] : ptr2[gz], in.dims[2]);
        int l = p.strds[3] * trimIndex(s3 ? gw+p.offs[3] : ptr3[gw], in.dims[3]);
        // offset input and output pointers
        const T *src = (const T*)in.ptr + (i+j+k+l);
        T *dst = (T*)out.ptr +(gx*out.strides[0]+gy*out.strides[1]+ gz*out.strides[2]+gw*out.strides[3]);
        // set the output
        dst[0] = src[0];
    }
}

template<typename T>
void index(Param<T> out, CParam<T> in, const IndexKernelParam_t& p)
{
    const dim3 threads(THREADS_X, THREADS_Y);

    int blks_x = divup(out.dims[0], threads.x);
    int blks_y = divup(out.dims[1], threads.y);

    dim3 blocks(blks_x*out.dims[2], blks_y*out.dims[3]);

    const int maxBlocksY = cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize[1];
    blocks.z = divup(blocks.y, maxBlocksY);
    blocks.y = divup(blocks.y, blocks.z);

    CUDA_LAUNCH((indexKernel<T>), blocks, threads, out, in, p, blks_x, blks_y);

    POST_LAUNCH_CHECK();
}

}

}
