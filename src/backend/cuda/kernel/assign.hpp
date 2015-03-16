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
#include <math.hpp>
#include <debug_cuda.hpp>

namespace cuda
{

namespace kernel
{

static const dim_type THREADS_X = 32;
static const dim_type THREADS_Y =  8;

typedef struct {
    dim_type  offs[4];
    dim_type strds[4];
    bool     isSeq[4];
    uint*      ptr[4];
} AssignKernelParam_t;

template<typename T>
__global__
void AssignKernel(Param<T> out, CParam<T> in, const AssignKernelParam_t p,
                 const dim_type nBBS0, const dim_type nBBS1)
{
    // retrieve index pointers
    // these can be 0 where af_array indexer is not used
    const uint* ptr0 = p.ptr[0];
    const uint* ptr1 = p.ptr[1];
    const uint* ptr2 = p.ptr[2];
    const uint* ptr3 = p.ptr[3];
    // retrive booleans that tell us which indexer to use
    const bool s0 = p.isSeq[0];
    const bool s1 = p.isSeq[1];
    const bool s2 = p.isSeq[2];
    const bool s3 = p.isSeq[3];

    const dim_type gz = blockIdx.x/nBBS0;
    const dim_type gw = blockIdx.y/nBBS1;
    const dim_type gx = blockDim.x * (blockIdx.x - gz*nBBS0) + threadIdx.x;
    const dim_type gy = blockDim.y * (blockIdx.y - gw*nBBS1) + threadIdx.y;

    if (gx<in.dims[0] && gy<in.dims[1] && gz<in.dims[2] && gw<in.dims[3]) {
        // calculate pointer offsets for input
        dim_type i = p.strds[0] * trimIndex(s0 ? gx+p.offs[0] : ptr0[gx], out.dims[0]);
        dim_type j = p.strds[1] * trimIndex(s1 ? gy+p.offs[1] : ptr1[gy], out.dims[1]);
        dim_type k = p.strds[2] * trimIndex(s2 ? gz+p.offs[2] : ptr2[gz], out.dims[2]);
        dim_type l = p.strds[3] * trimIndex(s3 ? gw+p.offs[3] : ptr3[gw], out.dims[3]);
        // offset input and output pointers
        const T *src = (const T*)in.ptr + (gx*in.strides[0]+gy*in.strides[1]+ gz*in.strides[2]+gw*in.strides[3]);
        T *dst = (T*)out.ptr +(i+j+k+l);
        // set the output
        dst[0] = src[0];
    }
}

template<typename T>
void assign(Param<T> out, CParam<T> in, const AssignKernelParam_t& p)
{
    const dim3 threads(THREADS_X, THREADS_Y);

    dim_type blks_x = divup(in.dims[0], threads.x);
    dim_type blks_y = divup(in.dims[1], threads.y);

    dim3 blocks(blks_x*in.dims[2], blks_y*in.dims[3]);

    AssignKernel<T> <<<blocks, threads>>> (out, in, p, blks_x, blks_y);

    POST_LAUNCH_CHECK();
}

}

}
