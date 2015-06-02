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
#include "shared.hpp"

namespace cuda
{

namespace kernel
{

static const int MAX_MORPH_FILTER_LEN = 17;
// cFilter is used by both 2d morph and 3d morph
// Maximum kernel size supported for 2d morph is 19x19*8 = 2888
// Maximum kernel size supported for 3d morph is 7x7x7*8 = 2744
// We will declare a char array as __constant__ array and allocate
// size necessary to hold doubles of FILTER_LEN*FILTER_LEN
__constant__ char cFilter[MAX_MORPH_FILTER_LEN*MAX_MORPH_FILTER_LEN*sizeof(double)];

static const int THREADS_X = 16;
static const int THREADS_Y = 16;

static const int CUBE_X    =  8;
static const int CUBE_Y    =  8;
static const int CUBE_Z    =  8;

__forceinline__ __device__ int lIdx(int x, int y,
        int stride1, int stride0)
{
    return (y*stride1 + x*stride0);
}

__forceinline__ __device__ int clamp(int f, int a, int b)
{
    return max(a, min(f, b));
}

template<typename T>
inline __device__ void load2ShrdMem(T * shrd, const T * const in,
        int lx, int ly, int shrdStride,
        int dim0, int dim1,
        int gx, int gy,
        int inStride1, int inStride0)
{
    int gx_  = clamp(gx, 0, dim0-1);
    int gy_  = clamp(gy, 0, dim1-1);
    shrd[ lIdx(lx, ly, shrdStride, 1) ] = in[ lIdx(gx_, gy_, inStride1, inStride0) ];
}

// kernel assumes mask/filter is square and hence does the
// necessary operations accordingly.
template<typename T, bool isDilation, int windLen>
static __global__ void morphKernel(Param<T> out, CParam<T> in,
                                   int nBBS0, int nBBS1)
{
    // get shared memory pointer
    SharedMemory<T> shared;
    T * shrdMem = shared.getPointer();

    // calculate necessary offset and window parameters
    const int halo   = windLen/2;
    const int padding= 2*halo;
    const int shrdLen= blockDim.x + padding + 1;

    // gfor batch offsets
    unsigned b2 = blockIdx.x / nBBS0;
    unsigned b3 = blockIdx.y / nBBS1;
    const T* iptr    = (const T *) in.ptr + (b2 *  in.strides[2] + b3 *  in.strides[3]);
    T*       optr    = (T *      )out.ptr + (b2 * out.strides[2] + b3 * out.strides[3]);

    int gx, gy, i, j;
    { //scoping out unnecessary variables
    // local neighborhood indices
    const int lx = threadIdx.x;
    const int ly = threadIdx.y;

    // global indices
    gx = blockDim.x * (blockIdx.x-b2*nBBS0) + lx;
    gy = blockDim.y * (blockIdx.y-b3*nBBS1) + ly;

    // offset values for pulling image to local memory
    int lx2      = lx + blockDim.x;
    int ly2      = ly + blockDim.y;
    int gx2      = gx + blockDim.x;
    int gy2      = gy + blockDim.y;

    // pull image to local memory
    load2ShrdMem(shrdMem, iptr, lx, ly, shrdLen,
                 in.dims[0], in.dims[1],
                 gx-halo, gy-halo,
                 in.strides[1], in.strides[0]);
    if (lx<padding) {
        load2ShrdMem(shrdMem, iptr, lx2, ly, shrdLen,
                     in.dims[0], in.dims[1],
                     gx2-halo, gy-halo,
                     in.strides[1], in.strides[0]);
    }
    if (ly<padding) {
        load2ShrdMem(shrdMem, iptr, lx, ly2, shrdLen,
                     in.dims[0], in.dims[1],
                     gx-halo, gy2-halo,
                     in.strides[1], in.strides[0]);
    }
    if (lx<padding && ly<padding) {
        load2ShrdMem(shrdMem, iptr, lx2, ly2, shrdLen,
                     in.dims[0], in.dims[1],
                     gx2-halo, gy2-halo,
                     in.strides[1], in.strides[0]);
    }
    i = lx + halo;
    j = ly + halo;
    }
    __syncthreads();

    const T * d_filt = (const T *)cFilter;
    T acc = shrdMem[ lIdx(i, j, shrdLen, 1) ];
#pragma unroll
    for(int wj=0; wj<windLen; ++wj) {
        int joff   = wj*windLen;
        int w_joff = (j+wj-halo)*shrdLen;
#pragma unroll
        for(int wi=0; wi<windLen; ++wi) {
            T cur  = shrdMem[w_joff + (i+wi-halo)];
            if (d_filt[joff+wi]) {
                if (isDilation)
                    acc = max(acc, cur);
                else
                    acc = min(acc, cur);
            }
        }
    }

    if (gx<in.dims[0] && gy<in.dims[1]) {
        int outIdx = lIdx(gx, gy, out.strides[1], out.strides[0]);
        optr[outIdx] = acc;
    }
}

__forceinline__ __device__ int lIdx3D(int x, int y, int z,
        int stride2, int stride1, int stride0)
{
    return (z*stride2 + y*stride1 + x*stride0);
}

template<typename T>
inline __device__ void load2ShrdVolume(T * shrd, const T * const in,
        int lx, int ly, int lz,
        int shrdStride1, int shrdStride2,
        int dim0, int dim1, int dim2,
        int gx, int gy, int gz,
        int inStride2, int inStride1, int inStride0)
{
    int gx_  = clamp(gx,0,dim0-1);
    int gy_  = clamp(gy,0,dim1-1);
    int gz_  = clamp(gz,0,dim2-1);
    int shrdIdx = lx + ly*shrdStride1 + lz*shrdStride2;
    int inIdx   = gx_*inStride0 + gy_*inStride1 + gz_*inStride2;
    shrd[ shrdIdx ] = in[ inIdx ];
}

// kernel assumes mask/filter is square and hence does the
// necessary operations accordingly.
template<typename T, bool isDilation, int windLen>
static __global__ void morph3DKernel(Param<T> out, CParam<T> in, int nBBS)
{
    // get shared memory pointer
    SharedMemory<T> shared;
    T * shrdMem = shared.getPointer();

    const int halo      = windLen/2;
    const int padding   = 2*halo;

    const int se_area   = windLen*windLen;
    const int shrdLen   = blockDim.x + padding + 1;
    const int shrdArea  = shrdLen * (blockDim.y+padding);

    // gfor batch offsets
    unsigned batchId = blockIdx.x / nBBS;

    const T* iptr    = (const T *) in.ptr + (batchId *  in.strides[3]);
    T*       optr    = (T *      )out.ptr + (batchId * out.strides[3]);

    int gx, gy, gz, i, j, k;
    { // scoping out unnecessary variables
    const int lx = threadIdx.x;
    const int ly = threadIdx.y;
    const int lz = threadIdx.z;

    gx = blockDim.x * (blockIdx.x-batchId*nBBS) + lx;
    gy = blockDim.y * blockIdx.y + ly;
    gz = blockDim.z * blockIdx.z + lz;

    const int gx2 = gx + blockDim.x;
    const int gy2 = gy + blockDim.y;
    const int gz2 = gz + blockDim.z;
    const int lx2 = lx + blockDim.x;
    const int ly2 = ly + blockDim.y;
    const int lz2 = lz + blockDim.z;

    // pull volume to shared memory
    load2ShrdVolume(shrdMem, iptr, lx, ly, lz, shrdLen, shrdArea,
                    in.dims[0], in.dims[1], in.dims[2],
                    gx-halo, gy-halo, gz-halo,
                    in.strides[2], in.strides[1], in.strides[0]);
    if (lx<padding) {
        load2ShrdVolume(shrdMem, iptr, lx2, ly, lz, shrdLen, shrdArea,
                        in.dims[0], in.dims[1], in.dims[2],
                        gx2-halo, gy-halo, gz-halo,
                        in.strides[2], in.strides[1], in.strides[0]);
    }
    if (ly<padding) {
        load2ShrdVolume(shrdMem, iptr, lx, ly2, lz, shrdLen, shrdArea,
                        in.dims[0], in.dims[1], in.dims[2],
                        gx-halo, gy2-halo, gz-halo,
                        in.strides[2], in.strides[1], in.strides[0]);
    }
    if (lz<padding) {
        load2ShrdVolume(shrdMem, iptr, lx, ly, lz2, shrdLen, shrdArea,
                        in.dims[0], in.dims[1], in.dims[2],
                        gx-halo, gy-halo, gz2-halo,
                        in.strides[2], in.strides[1], in.strides[0]);
    }
    if (lx<padding && ly<padding) {
        load2ShrdVolume(shrdMem, iptr, lx2, ly2, lz, shrdLen, shrdArea,
                        in.dims[0], in.dims[1], in.dims[2],
                        gx2-halo, gy2-halo, gz-halo,
                        in.strides[2], in.strides[1], in.strides[0]);
    }
    if (ly<padding && lz<padding) {
        load2ShrdVolume(shrdMem, iptr, lx, ly2, lz2, shrdLen, shrdArea,
                        in.dims[0], in.dims[1], in.dims[2],
                        gx-halo, gy2-halo, gz2-halo,
                        in.strides[2], in.strides[1], in.strides[0]);
    }
    if (lz<padding && lx<padding) {
        load2ShrdVolume(shrdMem, iptr, lx2, ly, lz2, shrdLen, shrdArea,
                        in.dims[0], in.dims[1], in.dims[2],
                        gx2-halo, gy-halo, gz2-halo,
                        in.strides[2], in.strides[1], in.strides[0]);
    }
    if (lx<padding && ly<padding && lz<padding) {
        load2ShrdVolume(shrdMem, iptr, lx2, ly2, lz2, shrdLen, shrdArea,
                        in.dims[0], in.dims[1], in.dims[2],
                        gx2-halo, gy2-halo, gz2-halo,
                        in.strides[2], in.strides[1], in.strides[0]);
    }
    __syncthreads();
    // indices of voxel owned by current thread
    i  = lx + halo;
    j  = ly + halo;
    k  = lz + halo;
    }

    const T * d_filt = (const T *)cFilter;
    T acc = shrdMem[ lIdx3D(i, j, k, shrdArea, shrdLen, 1) ];
#pragma unroll
    for(int wk=0; wk<windLen; ++wk) {
        int koff   = wk*se_area;
        int w_koff = (k+wk-halo)*shrdArea;
#pragma unroll
        for(int wj=0; wj<windLen; ++wj) {
        int joff   = wj*windLen;
        int w_joff = (j+wj-halo)*shrdLen;
#pragma unroll
            for(int wi=0; wi<windLen; ++wi) {
                T cur  = shrdMem[w_koff+w_joff + i+wi-halo];
                if (d_filt[koff+joff+wi]) {
                    if (isDilation)
                        acc = max(acc, cur);
                    else
                        acc = min(acc, cur);
                }
            }
        }
    }

    if (gx<in.dims[0] && gy<in.dims[1] && gz<in.dims[2]) {
        int outIdx = gz * out.strides[2] +
                          gy * out.strides[1] +
                          gx * out.strides[0];
        optr[outIdx] = acc;
    }
}

template<typename T, bool isDilation>
void morph(Param<T> out, CParam<T> in, int windLen)
{
    dim3 threads(kernel::THREADS_X, kernel::THREADS_Y);

    int blk_x = divup(in.dims[0], THREADS_X);
    int blk_y = divup(in.dims[1], THREADS_Y);
    // launch batch * blk_x blocks along x dimension
    dim3 blocks(blk_x * in.dims[2], blk_y * in.dims[3]);

    // calculate shared memory size
    int halo      = windLen/2;
    int padding   = 2*halo;
    int shrdLen   = kernel::THREADS_X + padding + 1; // +1 for to avoid bank conflicts
    int shrdSize  = shrdLen * (kernel::THREADS_Y + padding) * sizeof(T);

    switch(windLen) {
        case  3: morphKernel<T, isDilation, 3> <<< blocks, threads, shrdSize>>>(out, in, blk_x, blk_y); break;
        case  5: morphKernel<T, isDilation, 5> <<< blocks, threads, shrdSize>>>(out, in, blk_x, blk_y); break;
        case  7: morphKernel<T, isDilation, 7> <<< blocks, threads, shrdSize>>>(out, in, blk_x, blk_y); break;
        case  9: morphKernel<T, isDilation, 9> <<< blocks, threads, shrdSize>>>(out, in, blk_x, blk_y); break;
        case 11: morphKernel<T, isDilation,11> <<< blocks, threads, shrdSize>>>(out, in, blk_x, blk_y); break;
        case 13: morphKernel<T, isDilation,13> <<< blocks, threads, shrdSize>>>(out, in, blk_x, blk_y); break;
        case 15: morphKernel<T, isDilation,15> <<< blocks, threads, shrdSize>>>(out, in, blk_x, blk_y); break;
        case 17: morphKernel<T, isDilation,17> <<< blocks, threads, shrdSize>>>(out, in, blk_x, blk_y); break;
        case 19: morphKernel<T, isDilation,19> <<< blocks, threads, shrdSize>>>(out, in, blk_x, blk_y); break;
        default: morphKernel<T, isDilation, 3> <<< blocks, threads, shrdSize>>>(out, in, blk_x, blk_y); break;
    }

    POST_LAUNCH_CHECK();
}

template<typename T, bool isDilation>
void morph3d(Param<T> out, CParam<T> in, int windLen)
{
    dim3 threads(kernel::CUBE_X, kernel::CUBE_Y, kernel::CUBE_Z);

    int blk_x = divup(in.dims[0], CUBE_X);
    int blk_y = divup(in.dims[1], CUBE_Y);
    int blk_z = divup(in.dims[2], CUBE_Z);
    dim3 blocks(blk_x * in.dims[3], blk_y, blk_z);

    // calculate shared memory size
    int halo      = windLen/2;
    int padding   = 2*halo;
    int shrdLen   = kernel::CUBE_X + padding + 1; // +1 for to avoid bank conflicts
    int shrdSize  = shrdLen * (kernel::CUBE_Y + padding) * (kernel::CUBE_Z + padding) * sizeof(T);

    switch(windLen) {
        case  3: morph3DKernel<T, isDilation, 3> <<< blocks, threads, shrdSize>>>(out, in, blk_x); break;
        case  5: morph3DKernel<T, isDilation, 5> <<< blocks, threads, shrdSize>>>(out, in, blk_x); break;
        case  7: morph3DKernel<T, isDilation, 7> <<< blocks, threads, shrdSize>>>(out, in, blk_x); break;
        default: morph3DKernel<T, isDilation, 3> <<< blocks, threads, shrdSize>>>(out, in, blk_x); break;
    }

    POST_LAUNCH_CHECK();
}

}
}
