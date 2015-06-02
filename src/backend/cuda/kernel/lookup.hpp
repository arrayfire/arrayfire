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
#include <utility.hpp>
#include <debug_cuda.hpp>

namespace cuda
{

namespace kernel
{

static const int THREADS = 256;

static const int THREADS_X = 32;
static const int THREADS_Y = 8;

static const int THRD_LOAD = THREADS_X/THREADS_Y;

template<typename in_t, typename idx_t>
__global__
void lookup1D(Param<in_t> out, CParam<in_t> in, CParam<idx_t> indices, int vDim)
{
    int idx = threadIdx.x + blockIdx.x * THREADS * THRD_LOAD;

    const in_t* inPtr   = (const in_t*)in.ptr;
    const idx_t* idxPtr = (const idx_t*)indices.ptr;

    in_t* outPtr  = (in_t*)out.ptr;

    int en = min(out.dims[vDim], idx + THRD_LOAD * THREADS);

    for (int oIdx = idx; oIdx < en; oIdx += THREADS) {
        int iIdx = trimIndex(idxPtr[oIdx], in.dims[vDim]);
        outPtr[oIdx] = inPtr[iIdx];
    }
}

template<typename in_t, typename idx_t, unsigned dim>
__global__
void lookupND(Param<in_t> out, CParam<in_t> in, CParam<idx_t> indices,
                    int nBBS0, int nBBS1)
{
    int lx = threadIdx.x;
    int ly = threadIdx.y;

    int gz = blockIdx.x/nBBS0;
    int gw = blockIdx.y/nBBS1;

    int gx = blockDim.x * (blockIdx.x - gz*nBBS0) + lx;
    int gy = blockDim.y * (blockIdx.y - gw*nBBS1) + ly;

    const idx_t *idxPtr = (const idx_t*)indices.ptr;

    int i = in.strides[0]*(dim==0 ? trimIndex((int)idxPtr[gx], in.dims[0]): gx);
    int j = in.strides[1]*(dim==1 ? trimIndex((int)idxPtr[gy], in.dims[1]): gy);
    int k = in.strides[2]*(dim==2 ? trimIndex((int)idxPtr[gz], in.dims[2]): gz);
    int l = in.strides[3]*(dim==3 ? trimIndex((int)idxPtr[gw], in.dims[3]): gw);

    const in_t *inPtr = (const in_t*)in.ptr + (i+j+k+l);
    in_t *outPtr = (in_t*)out.ptr +(gx*out.strides[0]+gy*out.strides[1]+
                                    gz*out.strides[2]+gw*out.strides[3]);

    if (gx<out.dims[0] && gy<out.dims[1] && gz<out.dims[2] && gw<out.dims[3]) {
        outPtr[0] = inPtr[0];
    }
}

template<typename in_t, typename idx_t, unsigned dim>
void lookup(Param<in_t> out, CParam<in_t> in, CParam<idx_t> indices, int nDims)
{
    if (nDims==1) {
        const dim3 threads(THREADS, 1);
        /* find which dimension has non-zero # of elements */
        int vDim = 0;
        for (int i=0; i<4; i++) {
            if (in.dims[i]==1)
                vDim++;
            else
                break;
        }

        int blks = divup(out.dims[vDim], THREADS*THRD_LOAD);

        dim3 blocks(blks, 1);

        lookup1D<in_t, idx_t> <<<blocks, threads>>> (out, in, indices, vDim);
    } else {
        const dim3 threads(THREADS_X, THREADS_Y);

        int blks_x = divup(out.dims[0], threads.x);
        int blks_y = divup(out.dims[1], threads.y);

        dim3 blocks(blks_x*out.dims[2], blks_y*out.dims[3]);

        lookupND<in_t, idx_t, dim> <<<blocks, threads>>> (out, in, indices, blks_x, blks_y);
    }

    POST_LAUNCH_CHECK();
}

}

}
