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

static const dim_type MAX_MEDFILTER_LEN = 15;

static const dim_type THREADS_X = 16;
static const dim_type THREADS_Y = 16;


// Exchange trick: Morgan McGuire, ShaderX 2008
#define swap(a,b)    { T tmp = a; a = min(a,b); b = max(tmp,b); }

__forceinline__ __device__
dim_type lIdx(dim_type x, dim_type y, dim_type stride1, dim_type stride0)
{
    return (y*stride1 + x*stride0);
}

template<typename T, af_pad_type pad>
__device__
void load2ShrdMem(T * shrd, const T * in,
                  dim_type lx, dim_type ly, dim_type shrdStride,
                  dim_type dim0, dim_type dim1,
                  dim_type gx, dim_type gy,
                  dim_type inStride1, dim_type inStride0)
{
    switch(pad) {
        case AF_ZERO:
            {
                if (gx<0 || gx>=dim0 || gy<0 || gy>=dim1)
                    shrd[lIdx(lx, ly, shrdStride, 1)] = T(0);
                else
                    shrd[lIdx(lx, ly, shrdStride, 1)] = in[lIdx(gx, gy, inStride1, inStride0)];
            }
            break;
        case AF_SYMMETRIC:
            {
                if (gx<0) gx *= -1;
                if (gy<0) gy *= -1;
                if (gx>=dim0) gx = 2*(dim0-1) - gx;
                if (gy>=dim1) gy = 2*(dim1-1) - gy;

                shrd[lIdx(lx, ly, shrdStride, 1)] = in[lIdx(gx, gy, inStride1, inStride0)];
            }
            break;
    }
}

template<typename T, af_pad_type pad, unsigned w_len, unsigned w_wid>
__global__
void medfilt(Param<T> out, CParam<T> in, dim_type nonBatchBlkSize)
{
    __shared__ T shrdMem[(THREADS_X+w_len-1)*(THREADS_Y+w_wid-1)];

    // calculate necessary offset and window parameters
    const dim_type padding = w_len-1;
    const dim_type halo    = padding/2;
    const dim_type shrdLen = blockDim.x + padding;

    // batch offsets
    unsigned batchId = blockIdx.x / nonBatchBlkSize;
    const T* iptr    = (const T *) in.ptr + (batchId *  in.strides[2]);
    T*       optr    = (T *      )out.ptr + (batchId * out.strides[2]);

    // local neighborhood indices
    dim_type lx = threadIdx.x;
    dim_type ly = threadIdx.y;

    // global indices
    dim_type gx = blockDim.x * (blockIdx.x-batchId*nonBatchBlkSize) + lx;
    dim_type gy = blockDim.y * blockIdx.y + ly;

    // offset values for pulling image to local memory
    dim_type lx2 = lx + blockDim.x;
    dim_type ly2 = ly + blockDim.y;
    dim_type gx2 = gx + blockDim.x;
    dim_type gy2 = gy + blockDim.y;

    // pull image to local memory
    load2ShrdMem<T, pad>(shrdMem, iptr, lx, ly, shrdLen,
                         in.dims[0], in.dims[1],
                         gx-halo, gy-halo,
                         in.strides[1], in.strides[0]);
    if (lx<padding) {
        load2ShrdMem<T, pad>(shrdMem, iptr, lx2, ly, shrdLen,
                             in.dims[0], in.dims[1],
                             gx2-halo, gy-halo,
                             in.strides[1], in.strides[0]);
    }
    if (ly<padding) {
        load2ShrdMem<T, pad>(shrdMem, iptr, lx, ly2, shrdLen,
                             in.dims[0], in.dims[1],
                             gx-halo, gy2-halo,
                             in.strides[1], in.strides[0]);
    }
    if (lx<padding && ly<padding) {
        load2ShrdMem<T, pad>(shrdMem, iptr, lx2, ly2, shrdLen,
                             in.dims[0], in.dims[1],
                             gx2-halo, gy2-halo,
                             in.strides[1], in.strides[0]);
    }
    __syncthreads();

    // Only continue if we're at a valid location
    if (gx < in.dims[0] && gy < in.dims[1]) {

        const dim_type ARR_SIZE = w_len * (w_wid-w_wid/2);
        // pull top half from shared memory into local memory
        T v[ARR_SIZE];
#pragma unroll
        for(dim_type k = 0; k <= w_wid/2; k++) {
#pragma unroll
            for(dim_type i = 0; i < w_len; i++) {
                v[w_len*k + i] = shrdMem[lIdx(lx+i,ly+k,shrdLen,1)];
            }
        }

        // with each pass, remove min and max values and add new value
        // initial sort
        // ensure min in first half, max in second half
#pragma unroll
        for(dim_type i = 0; i < ARR_SIZE/2; i++) {
            swap(v[i], v[ARR_SIZE-1-i]);
        }
        // move min in first half to first pos
#pragma unroll
        for(dim_type i = 1; i < (ARR_SIZE+1)/2; i++) {
            swap(v[0], v[i]);
        }
        // move max in second half to last pos
#pragma unroll
        for(dim_type i = ARR_SIZE-2; i >= ARR_SIZE/2; i--) {
            swap(v[i], v[ARR_SIZE-1]);
        }

        dim_type last = ARR_SIZE-1;

        for(dim_type k = 1+w_wid/2; k < w_wid; k++) {

            for(dim_type j = 0; j < w_len; j++) {

                // add new contestant to first position in array
                v[0] = shrdMem[lIdx(lx+j, ly+k, shrdLen, 1)];

                last--;

                // place max in last half, min in first half
                for(dim_type i = 0; i < (last+1)/2; i++) {
                    swap(v[i], v[last-i]);
                }
                // now perform swaps on each half such that
                // max is in last pos, min is in first pos
                for(dim_type i = 1; i <= last/2; i++) {
                    swap(v[0], v[i]);
                }
                for(dim_type i = last-1; i >= (last+1)/2; i--) {
                    swap(v[i], v[last]);
                }
            }
        }

        // no more new contestants
        // may still have to sort the last row
        // each outer loop drops the min and max
        for(dim_type k = 1; k < w_len/2; k++) {
            // move max/min into respective halves
            for(dim_type i = k; i < w_len/2; i++) {
                swap(v[i], v[w_len-1-i]);
            }
            // move min into first pos
            for(dim_type i = k+1; i <= w_len/2; i++) {
                swap(v[k], v[i]);
            }
            // move max into last pos
            for(dim_type i = w_len-k-2; i >= w_len/2; i--) {
                swap(v[i], v[w_len-1-k]);
            }
        }

        // pick the middle element of the first row
        optr[gy*out.strides[1]+gx*out.strides[0]] = v[w_len/2];
    }
}

template<typename T, af_pad_type pad>
void medfilt(Param<T> out, CParam<T> in, dim_type w_len, dim_type w_wid)
{
    const dim3 threads(THREADS_X, THREADS_Y);

    dim_type blk_x = divup(in.dims[0], threads.x);
    dim_type blk_y = divup(in.dims[1], threads.y);

    dim3 blocks(blk_x*in.dims[2], blk_y);

    switch(w_len) {
        case  3: (medfilt<T, pad,  3,  3>)<<<blocks, threads>>>(out, in, blk_x); break;
        case  5: (medfilt<T, pad,  5,  5>)<<<blocks, threads>>>(out, in, blk_x); break;
        case  7: (medfilt<T, pad,  7,  7>)<<<blocks, threads>>>(out, in, blk_x); break;
        case  9: (medfilt<T, pad,  9,  9>)<<<blocks, threads>>>(out, in, blk_x); break;
        case 11: (medfilt<T, pad, 11, 11>)<<<blocks, threads>>>(out, in, blk_x); break;
        case 13: (medfilt<T, pad, 13, 13>)<<<blocks, threads>>>(out, in, blk_x); break;
        case 15: (medfilt<T, pad, 15, 15>)<<<blocks, threads>>>(out, in, blk_x); break;
    }

    POST_LAUNCH_CHECK();
}

}

}
