/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <platform.hpp>
#include <backend.hpp>
#include <common/dispatch.hpp>
#include <Param.hpp>
#include <debug_cuda.hpp>
#include "shared.hpp"

namespace cuda
{

namespace kernel
{

static const int MAX_MEDFILTER1_LEN = 121;
static const int MAX_MEDFILTER2_LEN = 15;

static const int THREADS_X = 16;
static const int THREADS_Y = 16;


// Exchange trick: Morgan McGuire, ShaderX 2008
#define swap(a,b)    { T tmp = a; a = min(a,b); b = max(tmp,b); }

__forceinline__ __device__
int lIdx(int x, int y, int stride1, int stride0)
{
    return (y*stride1 + x*stride0);
}

template<typename T, af_border_type pad>
__device__
void load2ShrdMem(T * shrd, const T * in,
                  int lx, int ly, int shrdStride,
                  int dim0, int dim1,
                  int gx, int gy,
                  int inStride1, int inStride0)
{
    switch(pad) {
        case AF_PAD_ZERO:
            {
                if (gx<0 || gx>=dim0 || gy<0 || gy>=dim1)
                    shrd[lIdx(lx, ly, shrdStride, 1)] = T(0);
                else
                    shrd[lIdx(lx, ly, shrdStride, 1)] = in[lIdx(gx, gy, inStride1, inStride0)];
            }
            break;
        case AF_PAD_SYM:
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

template<typename T, af_border_type pad>
__device__
void load2ShrdMem_1d(T * shrd, const T * in,
                    int lx, int dim0, int gx, int inStride0)
{
    switch(pad) {
        case AF_PAD_ZERO:
        {
            if (gx<0 || gx>=dim0)
                shrd[lx] = T(0);
            else
                shrd[lx] = in[gx];
        }
        break;
        case AF_PAD_SYM:
        {
            if (gx<0) gx *= -1;
            if (gx>=dim0) gx = 2*(dim0-1) - gx;

            shrd[lx] = in[gx];
        }
        break;
    }
}

template<typename T, af_border_type pad, unsigned w_len, unsigned w_wid>
__global__
void medfilt2(Param<T> out, CParam<T> in, int nBBS0, int nBBS1)
{
    __shared__ T shrdMem[(THREADS_X+w_len-1)*(THREADS_Y+w_wid-1)];

    // calculate necessary offset and window parameters
    const int padding = w_len-1;
    const int halo    = padding/2;
    const int shrdLen = blockDim.x + padding;

    // batch offsets
    unsigned b2 = blockIdx.x / nBBS0;
    unsigned b3 = blockIdx.y / nBBS1;
    const T* iptr    = (const T *) in.ptr + (b2 *  in.strides[2] + b3 *  in.strides[3]);
    T*       optr    = (T *      )out.ptr + (b2 * out.strides[2] + b3 * out.strides[3]);

    // local neighborhood indices
    int lx = threadIdx.x;
    int ly = threadIdx.y;

    // global indices
    int gx = blockDim.x * (blockIdx.x-b2*nBBS0) + lx;
    int gy = blockDim.y * (blockIdx.y-b3*nBBS1) + ly;

    // pull image to local memory
    for (int b=ly, gy2=gy; b<shrdLen; b+=blockDim.y, gy2+=blockDim.y) {
        // move row_set get_local_size(1) along coloumns
        for (int a=lx, gx2=gx; a<shrdLen; a+=blockDim.x, gx2+=blockDim.x) {
            load2ShrdMem<T, pad>(shrdMem, iptr, a, b, shrdLen, in.dims[0], in.dims[1],
                    gx2-halo, gy2-halo, in.strides[1], in.strides[0]);
        }
    }

    __syncthreads();

    // Only continue if we're at a valid location
    if (gx < in.dims[0] && gy < in.dims[1]) {

        const int ARR_SIZE = w_len * (w_wid-w_wid/2);
        // pull top half from shared memory into local memory
        T v[ARR_SIZE];
#pragma unroll
        for(int k = 0; k <= w_wid/2; k++) {
#pragma unroll
            for(int i = 0; i < w_len; i++) {
                v[w_len*k + i] = shrdMem[lIdx(lx+i,ly+k,shrdLen,1)];
            }
        }

        // with each pass, remove min and max values and add new value
        // initial sort
        // ensure min in first half, max in second half
#pragma unroll
        for(int i = 0; i < ARR_SIZE/2; i++) {
            swap(v[i], v[ARR_SIZE-1-i]);
        }
        // move min in first half to first pos
#pragma unroll
        for(int i = 1; i < (ARR_SIZE+1)/2; i++) {
            swap(v[0], v[i]);
        }
        // move max in second half to last pos
#pragma unroll
        for(int i = ARR_SIZE-2; i >= ARR_SIZE/2; i--) {
            swap(v[i], v[ARR_SIZE-1]);
        }

        int last = ARR_SIZE-1;

        for(int k = 1+w_wid/2; k < w_wid; k++) {

            for(int j = 0; j < w_len; j++) {

                // add new contestant to first position in array
                v[0] = shrdMem[lIdx(lx+j, ly+k, shrdLen, 1)];

                last--;

                // place max in last half, min in first half
                for(int i = 0; i < (last+1)/2; i++) {
                    swap(v[i], v[last-i]);
                }
                // now perform swaps on each half such that
                // max is in last pos, min is in first pos
                for(int i = 1; i <= last/2; i++) {
                    swap(v[0], v[i]);
                }
                for(int i = last-1; i >= (last+1)/2; i--) {
                    swap(v[i], v[last]);
                }
            }
        }

        // no more new contestants
        // may still have to sort the last row
        // each outer loop drops the min and max
        for(int k = 1; k < w_len/2; k++) {
            // move max/min into respective halves
            for(int i = k; i < w_len/2; i++) {
                swap(v[i], v[w_len-1-i]);
            }
            // move min into first pos
            for(int i = k+1; i <= w_len/2; i++) {
                swap(v[k], v[i]);
            }
            // move max into last pos
            for(int i = w_len-k-2; i >= w_len/2; i--) {
                swap(v[i], v[w_len-1-k]);
            }
        }

        // pick the middle element of the first row
        optr[gy*out.strides[1]+gx*out.strides[0]] = v[w_len/2];
    }
}

template<typename T, af_border_type pad, unsigned ARR_SIZE>
__global__
void medfilt1(Param<T> out, CParam<T> in, unsigned w_wid, int nBBS0)
{
    SharedMemory<T> shared;
    T * shrdMem = shared.getPointer();

    // calculate necessary offset and window parameters
    const int padding = w_wid-1;
    const int halo    = padding/2;
    const int shrdLen = blockDim.x + padding;

    // batch offsets
    unsigned b1 = blockIdx.x / nBBS0;
    unsigned b2 = blockIdx.y;
    unsigned b3 = blockIdx.z;

    const T* iptr    = (const T *) in.ptr + (b1 * in.strides[1] + b2 *  in.strides[2] + b3 *  in.strides[3]);
    T*       optr    = (T *      )out.ptr + (b1 * in.strides[1] + b2 * out.strides[2] + b3 * out.strides[3]);

    // local neighborhood indices
    int lx = threadIdx.x;

    // global indices
    int gx = blockDim.x * (blockIdx.x - b1 * nBBS0) + lx;

    // pull signal to local memory
    for (int a=lx, gx2=gx; a<shrdLen; a+=blockDim.x, gx2+=blockDim.x) {
        load2ShrdMem_1d<T, pad>(shrdMem, iptr, a, in.dims[0], gx2-halo, in.strides[0]);
    }

    __syncthreads();

    // Only continue if we're at a valid location
    if (gx < in.dims[0]) {
        const int ARR_BOUNDARY = (w_wid-w_wid/2) + 1;
        // pull top half from shared memory into local memory
        T v[ARR_SIZE];

#pragma unroll
        for(int k = 0; k <= w_wid/2 + 1; k++) {
            v[k] = shrdMem[lx+k];
        }
        // with each pass, remove min and max values and add new value
        // initial sort
        // ensure min in first half, max in second half
#pragma unroll
        for(int i = 0; i < ARR_BOUNDARY/2; i++) {
            swap(v[i], v[ARR_BOUNDARY-1-i]);
        }
        // move min in first half to first pos
#pragma unroll
        for(int i = 1; i < (ARR_BOUNDARY+1)/2; i++) {
            swap(v[0], v[i]);
        }
        // move max in second half to last pos
#pragma unroll
        for(int i = ARR_BOUNDARY-2; i >= ARR_BOUNDARY/2; i--) {
            swap(v[i], v[ARR_BOUNDARY-1]);
        }

        int last = ARR_BOUNDARY-1;

        for(int k = w_wid/2 + 2; k < w_wid; k++) {
            // add new contestant to first position in array
            v[0] = shrdMem[lx + k];

            last--;

            // place max in last half, min in first half
            for(int i = 0; i < (last+1)/2; i++) {
                swap(v[i], v[last-i]);
            }
            // now perform swaps on each half such that
            // max is in last pos, min is in first pos
            for(int i = 1; i <= last/2; i++) {
                swap(v[0], v[i]);
            }
            for(int i = last-1; i >= (last+1)/2; i--) {
                swap(v[i], v[last]);
            }
        }

        // no more new contestants
        // may still have to sort the last row
        // each outer loop drops the min and max
        for(int k = 0; k < last; k++) {
            // move max/min into respective halves
            for(int i = k; i < ARR_BOUNDARY/2; i++) {
                swap(v[i], v[ARR_BOUNDARY-1-i]);
            }
            // move min into first pos
            for(int i = k+1; i <= ARR_BOUNDARY/2; i++) {
                swap(v[k], v[i]);
            }
            // move max into last pos
            for(int i = ARR_BOUNDARY-k-2; i >= ARR_BOUNDARY/2; i--) {
                swap(v[i], v[ARR_BOUNDARY-1-k]);
            }
        }

        // pick the middle element of the first row
        optr[gx*out.strides[0]] = v[last/2];
    }
}

template<typename T, af_border_type pad>
void medfilt2(Param<T> out, CParam<T> in, int w_len, int w_wid)
{
    const dim3 threads(THREADS_X, THREADS_Y);

    int blk_x = divup(in.dims[0], threads.x);
    int blk_y = divup(in.dims[1], threads.y);

    dim3 blocks(blk_x*in.dims[2], blk_y*in.dims[3]);

    switch(w_len) {
        case  3: CUDA_LAUNCH((medfilt2<T,pad, 3, 3>), blocks, threads, out, in, blk_x, blk_y); break;
        case  5: CUDA_LAUNCH((medfilt2<T,pad, 5, 5>), blocks, threads, out, in, blk_x, blk_y); break;
        case  7: CUDA_LAUNCH((medfilt2<T,pad, 7, 7>), blocks, threads, out, in, blk_x, blk_y); break;
        case  9: CUDA_LAUNCH((medfilt2<T,pad, 9, 9>), blocks, threads, out, in, blk_x, blk_y); break;
        case 11: CUDA_LAUNCH((medfilt2<T,pad,11,11>), blocks, threads, out, in, blk_x, blk_y); break;
        case 13: CUDA_LAUNCH((medfilt2<T,pad,13,13>), blocks, threads, out, in, blk_x, blk_y); break;
        case 15: CUDA_LAUNCH((medfilt2<T,pad,15,15>), blocks, threads, out, in, blk_x, blk_y); break;
    }

    POST_LAUNCH_CHECK();
}

template<typename T, af_border_type pad>
void medfilt1(Param<T> out, CParam<T> in, int w_wid)
{
    const dim3 threads(THREADS_X);

    int blk_x = divup(in.dims[0], threads.x);

    dim3 blocks(blk_x*in.dims[1], in.dims[2], in.dims[3] );

    const size_t shrdMemBytes = sizeof(T) * (THREADS_X + w_wid - 1);

    switch(w_wid) {
        case  3: CUDA_LAUNCH_SMEM((medfilt1<T,pad, 3>), blocks, threads, shrdMemBytes, out, in, w_wid, blk_x);
        case  5: CUDA_LAUNCH_SMEM((medfilt1<T,pad, 4>), blocks, threads, shrdMemBytes, out, in, w_wid, blk_x);
        case  7: CUDA_LAUNCH_SMEM((medfilt1<T,pad, 5>), blocks, threads, shrdMemBytes, out, in, w_wid, blk_x);
        case  9: CUDA_LAUNCH_SMEM((medfilt1<T,pad, 6>), blocks, threads, shrdMemBytes, out, in, w_wid, blk_x);
        case 11: CUDA_LAUNCH_SMEM((medfilt1<T,pad, 7>), blocks, threads, shrdMemBytes, out, in, w_wid, blk_x);
        case 13: CUDA_LAUNCH_SMEM((medfilt1<T,pad, 8>), blocks, threads, shrdMemBytes, out, in, w_wid, blk_x);
        case 15: CUDA_LAUNCH_SMEM((medfilt1<T,pad, 9>), blocks, threads, shrdMemBytes, out, in, w_wid, blk_x);
        case 17: CUDA_LAUNCH_SMEM((medfilt1<T,pad,10>), blocks, threads, shrdMemBytes, out, in, w_wid, blk_x);
        case 19: CUDA_LAUNCH_SMEM((medfilt1<T,pad,11>), blocks, threads, shrdMemBytes, out, in, w_wid, blk_x);
        default: CUDA_LAUNCH_SMEM((medfilt1<T,pad,62>), blocks, threads, shrdMemBytes, out, in, w_wid, blk_x);
    }

    POST_LAUNCH_CHECK();
}

}

}
