/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <limits>
#include <backend.hpp>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_cuda.hpp>
#include <math.hpp>
#include <ops.hpp>
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

template<typename T, bool isDilation>
inline __device__ void load2ShrdMem(T * shrd, const T * const in,
        int lx, int ly, int shrdStride,
        int dim0, int dim1,
        int gx, int gy,
        int inStride1, int inStride0)
{
    T val = isDilation ? Binary<T, af_max_t>().init() : Binary<T, af_min_t>().init();
    if (gx>=0 && gx<dim0 && gy>=0 && gy<dim1) {
        val = in[ lIdx(gx, gy, inStride1, inStride0) ];
    }
    shrd[ lIdx(lx, ly, shrdStride, 1) ] = val;
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
    const int halo     = windLen/2;
    const int padding  = (windLen%2==0 ? (windLen-1) : (2*(windLen/2)));
    const int shrdLen  = blockDim.x + padding + 1;
    const int shrdLen1 = blockDim.y + padding;

    // gfor batch offsets
    unsigned b2 = blockIdx.x / nBBS0;
    unsigned b3 = blockIdx.y / nBBS1;
    const T* iptr    = (const T *) in.ptr + (b2 *  in.strides[2] + b3 *  in.strides[3]);
    T*       optr    = (T *      )out.ptr + (b2 * out.strides[2] + b3 * out.strides[3]);

    const int lx = threadIdx.x;
    const int ly = threadIdx.y;

    // global indices
    const int gx = blockDim.x * (blockIdx.x-b2*nBBS0) + lx;
    const int gy = blockDim.y * (blockIdx.y-b3*nBBS1) + ly;

    // pull image to local memory
    for (int b=ly, gy2=gy; b<shrdLen1; b+=blockDim.y, gy2+=blockDim.y) {
        // move row_set get_local_size(1) along coloumns
        for (int a=lx, gx2=gx; a<shrdLen; a+=blockDim.x, gx2+=blockDim.x) {
            load2ShrdMem<T, isDilation>(shrdMem, iptr, a, b, shrdLen,
                                        in.dims[0], in.dims[1],
                                        gx2-halo, gy2-halo, in.strides[1], in.strides[0]);
        }
    }

    int i = lx + halo;
    int j = ly + halo;

    __syncthreads();

    const T * d_filt = (const T *)cFilter;
    T acc = isDilation ? Binary<T, af_max_t>().init() : Binary<T, af_min_t>().init();
#pragma unroll
    for(int wj=0; wj<windLen; ++wj) {
        int joff   = wj*windLen;
        int w_joff = (j+wj-halo)*shrdLen;
#pragma unroll
        for(int wi=0; wi<windLen; ++wi) {
            if (d_filt[joff+wi] > (T)0) {
                T cur  = shrdMem[w_joff + (i+wi-halo)];
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

template<typename T, bool isDilation>
inline __device__ void load2ShrdVolume(T * shrd, const T * const in,
        int lx, int ly, int lz,
        int shrdStride1, int shrdStride2,
        int dim0, int dim1, int dim2,
        int gx, int gy, int gz,
        int inStride2, int inStride1, int inStride0)
{
    T val = isDilation ? Binary<T, af_max_t>().init() : Binary<T, af_min_t>().init();
    if (gx>=0 && gx<dim0 &&
        gy>=0 && gy<dim1 &&
        gz>=0 && gz<dim2) {
        val = in[gx*inStride0 + gy*inStride1 + gz*inStride2];
    }
    shrd[lx + ly*shrdStride1 + lz*shrdStride2] = val;
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
    const int padding   = (windLen%2==0 ? (windLen-1) : (2*(windLen/2)));

    const int se_area   = windLen*windLen;
    const int shrdLen   = blockDim.x + padding + 1;
    const int shrdLen1  = blockDim.y + padding;
    const int shrdLen2  = blockDim.z + padding;
    const int shrdArea  = shrdLen * shrdLen1;

    // gfor batch offsets
    unsigned batchId = blockIdx.x / nBBS;

    const T* iptr    = (const T *) in.ptr + (batchId *  in.strides[3]);
    T*       optr    = (T *      )out.ptr + (batchId * out.strides[3]);

    const int lx = threadIdx.x;
    const int ly = threadIdx.y;
    const int lz = threadIdx.z;

    const int gx = blockDim.x * (blockIdx.x-batchId*nBBS) + lx;
    const int gy = blockDim.y * blockIdx.y + ly;
    const int gz = blockDim.z * blockIdx.z + lz;

    for (int c=lz, gz2=gz; c<shrdLen2; c+=blockDim.z, gz2+=blockDim.z) {
        for (int b=ly, gy2=gy; b<shrdLen1; b+=blockDim.y, gy2+=blockDim.y) {
            for (int a=lx, gx2=gx; a<shrdLen; a+=blockDim.x, gx2+=blockDim.x) {
                load2ShrdVolume<T, isDilation>(shrdMem, iptr, a, b, c, shrdLen, shrdArea,
                                               in.dims[0], in.dims[1], in.dims[2],
                                               gx2-halo, gy2-halo, gz2-halo,
                                               in.strides[2], in.strides[1], in.strides[0]);
            }
        }
    }

    __syncthreads();
    // indices of voxel owned by current thread
    int i  = lx + halo;
    int j  = ly + halo;
    int k  = lz + halo;

    const T * d_filt = (const T *)cFilter;
    T acc = isDilation ? Binary<T, af_max_t>().init() : Binary<T, af_min_t>().init();
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
                if (d_filt[koff+joff+wi]) {
                    T cur  = shrdMem[w_koff+w_joff + i+wi-halo];
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
    int padding   = (windLen%2==0 ? (windLen-1) : (2*(windLen/2)));
    int shrdLen   = kernel::THREADS_X + padding + 1; // +1 for to avoid bank conflicts
    int shrdSize  = shrdLen * (kernel::THREADS_Y + padding) * sizeof(T);

    switch(windLen) {
        case  2: CUDA_LAUNCH_SMEM((morphKernel<T, isDilation,  2>), blocks, threads, shrdSize, out, in, blk_x, blk_y); break;
        case  3: CUDA_LAUNCH_SMEM((morphKernel<T, isDilation,  3>), blocks, threads, shrdSize, out, in, blk_x, blk_y); break;
        case  4: CUDA_LAUNCH_SMEM((morphKernel<T, isDilation,  4>), blocks, threads, shrdSize, out, in, blk_x, blk_y); break;
        case  5: CUDA_LAUNCH_SMEM((morphKernel<T, isDilation,  5>), blocks, threads, shrdSize, out, in, blk_x, blk_y); break;
        case  6: CUDA_LAUNCH_SMEM((morphKernel<T, isDilation,  6>), blocks, threads, shrdSize, out, in, blk_x, blk_y); break;
        case  7: CUDA_LAUNCH_SMEM((morphKernel<T, isDilation,  7>), blocks, threads, shrdSize, out, in, blk_x, blk_y); break;
        case  8: CUDA_LAUNCH_SMEM((morphKernel<T, isDilation,  8>), blocks, threads, shrdSize, out, in, blk_x, blk_y); break;
        case  9: CUDA_LAUNCH_SMEM((morphKernel<T, isDilation,  9>), blocks, threads, shrdSize, out, in, blk_x, blk_y); break;
        case 10: CUDA_LAUNCH_SMEM((morphKernel<T, isDilation, 10>), blocks, threads, shrdSize, out, in, blk_x, blk_y); break;
        case 11: CUDA_LAUNCH_SMEM((morphKernel<T, isDilation, 11>), blocks, threads, shrdSize, out, in, blk_x, blk_y); break;
        case 12: CUDA_LAUNCH_SMEM((morphKernel<T, isDilation, 12>), blocks, threads, shrdSize, out, in, blk_x, blk_y); break;
        case 13: CUDA_LAUNCH_SMEM((morphKernel<T, isDilation, 13>), blocks, threads, shrdSize, out, in, blk_x, blk_y); break;
        case 14: CUDA_LAUNCH_SMEM((morphKernel<T, isDilation, 14>), blocks, threads, shrdSize, out, in, blk_x, blk_y); break;
        case 15: CUDA_LAUNCH_SMEM((morphKernel<T, isDilation, 15>), blocks, threads, shrdSize, out, in, blk_x, blk_y); break;
        case 16: CUDA_LAUNCH_SMEM((morphKernel<T, isDilation, 16>), blocks, threads, shrdSize, out, in, blk_x, blk_y); break;
        case 17: CUDA_LAUNCH_SMEM((morphKernel<T, isDilation, 17>), blocks, threads, shrdSize, out, in, blk_x, blk_y); break;
        case 18: CUDA_LAUNCH_SMEM((morphKernel<T, isDilation, 18>), blocks, threads, shrdSize, out, in, blk_x, blk_y); break;
        case 19: CUDA_LAUNCH_SMEM((morphKernel<T, isDilation, 19>), blocks, threads, shrdSize, out, in, blk_x, blk_y); break;
        default: CUDA_NOT_SUPPORTED("Morph does not support kernels larger than 19.");
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
    int padding   = (windLen%2==0 ? (windLen-1) : (2*(windLen/2)));
    int shrdLen   = kernel::CUBE_X + padding + 1; // +1 for to avoid bank conflicts
    int shrdSize  = shrdLen * (kernel::CUBE_Y + padding) * (kernel::CUBE_Z + padding) * sizeof(T);

    switch(windLen) {
        case  2: CUDA_LAUNCH_SMEM((morph3DKernel<T, isDilation, 2>), blocks, threads, shrdSize, out, in, blk_x); break;
        case  3: CUDA_LAUNCH_SMEM((morph3DKernel<T, isDilation, 3>), blocks, threads, shrdSize, out, in, blk_x); break;
        case  4: CUDA_LAUNCH_SMEM((morph3DKernel<T, isDilation, 4>), blocks, threads, shrdSize, out, in, blk_x); break;
        case  5: CUDA_LAUNCH_SMEM((morph3DKernel<T, isDilation, 5>), blocks, threads, shrdSize, out, in, blk_x); break;
        case  6: CUDA_LAUNCH_SMEM((morph3DKernel<T, isDilation, 6>), blocks, threads, shrdSize, out, in, blk_x); break;
        case  7: CUDA_LAUNCH_SMEM((morph3DKernel<T, isDilation, 7>), blocks, threads, shrdSize, out, in, blk_x); break;
        default: CUDA_NOT_SUPPORTED("Morph 3D does not support kernels larger than 7.");
    }

    POST_LAUNCH_CHECK();
}
}
}
