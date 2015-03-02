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

namespace cuda
{

namespace kernel
{

static const dim_type THREADS = 256;

static const dim_type THREADS_X = 32;
static const dim_type THREADS_Y = 8;

static const dim_type THRD_LOAD = THREADS_X/THREADS_Y;

__device__
dim_type trimIndex(dim_type idx, const dim_type &len)
{
    dim_type ret_val = idx;
    dim_type offset  = abs(ret_val)%len;
    if (ret_val<0) {
        ret_val = offset-1;
    } else if (ret_val>=len) {
        ret_val = len-offset-1;
    }
    return ret_val;
}

template<typename in_t, typename idx_t>
__global__
void arrayIndex1D(Param<in_t> out, CParam<in_t> in, CParam<idx_t> indices, dim_type vDim)
{
    dim_type idx = (threadIdx.x + blockIdx.x * THREADS) * THRD_LOAD;

    const in_t* inPtr   = (const in_t*)in.ptr;
    const idx_t* idxPtr = (const idx_t*)indices.ptr;

    in_t* outPtr  = (in_t*)out.ptr;

    for (dim_type i=0; i<THRD_LOAD; i+=THREADS_Y) {
        dim_type oIdx = idx + i;
        if (oIdx < out.dims[vDim]) {
            dim_type iIdx = trimIndex(idxPtr[oIdx], in.dims[vDim]);
            outPtr[oIdx] = inPtr[iIdx];
        }
    }
}

template<typename in_t, typename idx_t, unsigned dim>
__global__
void arrayIndexND(Param<in_t> out, CParam<in_t> in, CParam<idx_t> indices,
                    dim_type nBBS0, dim_type nBBS1)
{
    dim_type lx = threadIdx.x;
    dim_type ly = threadIdx.y;

    dim_type gz = blockIdx.x/nBBS0;
    dim_type gw = blockIdx.y/nBBS1;

    dim_type gx = blockDim.x * (blockIdx.x - gz*nBBS0) + lx;
    dim_type gy = blockDim.y * (blockIdx.y - gw*nBBS1) + ly;

    const idx_t *idxPtr = (const idx_t*)indices.ptr;

    dim_type i = in.strides[0]*(dim==0 ? trimIndex((dim_type)idxPtr[gx], in.dims[0]): gx);
    dim_type j = in.strides[1]*(dim==1 ? trimIndex((dim_type)idxPtr[gy], in.dims[1]): gy);
    dim_type k = in.strides[2]*(dim==2 ? trimIndex((dim_type)idxPtr[gz], in.dims[2]): gz);
    dim_type l = in.strides[3]*(dim==3 ? trimIndex((dim_type)idxPtr[gw], in.dims[3]): gw);

    const in_t *inPtr = (const in_t*)in.ptr + (i+j+k+l);
    in_t *outPtr = (in_t*)out.ptr +(gx*out.strides[0]+gy*out.strides[1]+
                                    gz*out.strides[2]+gw*out.strides[3]);

    if (gx<out.dims[0] && gy<out.dims[1] && gz<out.dims[2] && gw<out.dims[3]) {
        outPtr[0] = inPtr[0];
    }
}

template<typename in_t, typename idx_t, unsigned dim>
void arrayIndex(Param<in_t> out, CParam<in_t> in, CParam<idx_t> indices, dim_type nDims)
{
    if (nDims==1) {
        const dim3 threads(THREADS, 1);
        /* find which dimension has non-zero # of elements */
        dim_type vDim = 0;
        for (dim_type i=0; i<4; i++) {
            if (in.dims[i]==1)
                vDim++;
            else
                break;
        }

        dim_type blks = divup(out.dims[vDim], threads.x*THRD_LOAD);

        dim3 blocks(blks, 1);

        arrayIndex1D<in_t, idx_t> <<<blocks, threads>>> (out, in, indices, vDim);
    } else {
        const dim3 threads(THREADS_X, THREADS_Y);

        dim_type blks_x = divup(out.dims[0], threads.x);
        dim_type blks_y = divup(out.dims[1], threads.y);

        dim3 blocks(blks_x*out.dims[2], blks_y*out.dims[3]);

        arrayIndexND<in_t, idx_t, dim> <<<blocks, threads>>> (out, in, indices, blks_x, blks_y);
    }

    POST_LAUNCH_CHECK();
}

}

}
