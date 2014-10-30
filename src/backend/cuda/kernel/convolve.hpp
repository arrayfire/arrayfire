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

static const dim_type THREADS   = 256;

static const dim_type THREADS_X = 16;
static const dim_type THREADS_Y = 16;

static const dim_type CUBE_X    =  8;
static const dim_type CUBE_Y    =  8;
static const dim_type CUBE_Z    =  4;

// below shared MAX_*_LEN's are calculated based on
// a maximum shared memory configuration of 48KB per block
// considering complex types as well
static const dim_type MAX_CONV1_FILTER_LEN = 129;
static const dim_type MAX_CONV2_FILTER_LEN = 17;
static const dim_type MAX_CONV3_FILTER_LEN = 5;

// we shall declare the maximum size required of above all three cases
// and re-use the same constant memory locations for every case
__constant__ char cFilter[2*(2ll*(MAX_CONV1_FILTER_LEN-1ll)+THREADS)*sizeof(double)];

__inline__ __device__
dim_type index(dim_type i, dim_type j, dim_type k, dim_type jstride, dim_type kstride)
{
    return i+j*jstride+k*kstride;
}

template<typename T>
__device__
T readSrc(T const *src, dim_type i, dim_type j, dim_type k, dim_type dims[], dim_type strides[])
{
    bool is_i = i>=0 && i<dims[0];
    bool is_j = j>=0 && j<dims[1];
    bool is_k = k>=0 && k<dims[2];
    if (is_i && is_j && is_k)
        return src[(i*strides[0] + j*strides[1] + k*strides[2])];
    else
        return scalar<T>(0);
}

template<typename T, typename accType, bool expand>
__global__
void convolve1(Param<T> out, CParam<T> signal, dim_type fLen, dim_type nonBatchBlkSize,
               dim_type oStep, dim_type sStep)
{
    SharedMemory<T> shared;

    T * shrdMem         = shared.getPointer();
    dim_type padding    = fLen-1;
    dim_type shrdLen    = blockDim.x + 2*padding;
    unsigned batchId    = blockIdx.x/nonBatchBlkSize;
    T *dst              = (T *)out.ptr          + oStep +(batchId*out.strides[1]);
    const T *src        = (const T *)signal.ptr + sStep +(batchId*signal.strides[1]);
    const T *impulse    = (const T *)cFilter;

    dim_type gx  = blockDim.x*(blockIdx.x-batchId*nonBatchBlkSize) + threadIdx.x;

    for (dim_type i=0; i<shrdLen; i+=blockDim.x) {
        dim_type idx = gx-padding + i;
        dim_type lx  = threadIdx.x+ i;
        if (lx<shrdLen)
            shrdMem[lx]  = (idx>=0 && idx<signal.dims[0]) ? src[idx*signal.strides[0]] : scalar<T>(0);
    }
    __syncthreads();

    if (gx>=0 && gx<out.dims[0]) {
        dim_type lx   = threadIdx.x + padding + (expand ? 0 : fLen/2);
        accType accum = scalar<accType>(0);
        for(dim_type f=0; f<fLen; ++f) {
            accum = accum + (shrdMem[lx-f]*impulse[f]);
        }
        dst[gx] = (T)accum;
    }
}

template<typename T, typename accType, bool expand>
__global__
void convolve2(Param<T> out, CParam<T> signal, dim_type fLen0, dim_type fLen1,
               dim_type nonBatchBlkSize, dim_type oStep, dim_type sStep)
{
    SharedMemory<T> shared;

    T * shrdMem       = shared.getPointer();
    dim_type pad0     = fLen0-1;
    dim_type pad1     = fLen1-1;
    dim_type shrdLen0 = blockDim.x + 2*pad0;
    unsigned batchId  = blockIdx.x/nonBatchBlkSize;
    T *dst            = (T *)out.ptr          + oStep + (batchId*out.strides[2]);
    const T *src      = (const T *)signal.ptr + sStep + (batchId*signal.strides[2]);
    const T *impulse  = (const T *)cFilter;

    dim_type lx = threadIdx.x;
    dim_type ly = threadIdx.y;
    dim_type gx = blockDim.x * (blockIdx.x-batchId*nonBatchBlkSize) + lx;
    dim_type gy = blockDim.y * blockIdx.y + ly;
    dim_type i = lx + pad0;
    dim_type j = ly + pad1;

    shrdMem[j*shrdLen0+i] = readSrc(src, gx, gy, 0, signal.dims, signal.strides);

    if (lx < pad0) {
        dim_type gx2 = gx + blockDim.x;
        dim_type lx2 = i  + blockDim.x;
        shrdMem[j*shrdLen0+ lx] = readSrc(src, gx-pad0, gy, 0, signal.dims, signal.strides);
        shrdMem[j*shrdLen0+lx2] = readSrc(src, gx2    , gy, 0, signal.dims, signal.strides);
    }
    if (ly < pad1) {
        dim_type gy2 = gy + blockDim.y;
        dim_type ly2 = j  + blockDim.y;
        shrdMem[ly*shrdLen0 +i] = readSrc(src, gx, gy-pad1, 0, signal.dims, signal.strides);
        shrdMem[ly2*shrdLen0+i] = readSrc(src, gx, gy2    , 0, signal.dims, signal.strides);
    }
    if (lx < pad0 && ly < pad1) {
        dim_type gx2 = gx + blockDim.x;
        dim_type lx2 = i  + blockDim.x;
        dim_type gy2 = gy + blockDim.y;
        dim_type ly2 = j  + blockDim.y;
        // 4 corner regions
        shrdMem[ly*shrdLen0+lx  ] = readSrc(src, gx-pad0, gy-pad1, 0, signal.dims, signal.strides);
        shrdMem[ly*shrdLen0+lx2 ] = readSrc(src, gx2    , gy-pad1, 0, signal.dims, signal.strides);
        shrdMem[ly2*shrdLen0+lx ] = readSrc(src, gx-pad0, gy2    , 0, signal.dims, signal.strides);
        shrdMem[ly2*shrdLen0+lx2] = readSrc(src, gx2    , gy2    , 0, signal.dims, signal.strides);
    }
    __syncthreads();

    if (gx>=0 && gx<out.dims[0] && gy>=0 && gy<out.dims[1]) {
        dim_type ci = i + (expand ? 0 : fLen0/2);
        dim_type cj = j + (expand ? 0 : fLen1/2);

        accType accum = scalar<accType>(0);
        for(dim_type fj=0; fj<fLen1; ++fj) {
            for(dim_type fi=0; fi<fLen0; ++fi) {
                T f_val = impulse[fj*fLen0+fi];
                T s_val = shrdMem[(cj-fj)*shrdLen0+(ci-fi)];
                accum   = accum + s_val*f_val;
            }
        }
        dst[gy*out.strides[1]+gx] = (T)accum;
    }
}

template<typename T, typename accType, bool expand>
__global__
void convolve3(Param<T> out, CParam<T> signal, dim_type fLen0, dim_type fLen1,
               dim_type fLen2, dim_type nonBatchBlkSize, dim_type oStep, dim_type sStep)
{
    SharedMemory<T> shared;

    T * shrdMem       = shared.getPointer();
    dim_type pad0     = fLen0-1;
    dim_type pad1     = fLen1-1;
    dim_type pad2     = fLen2-1;
    dim_type shrdLen0 = blockDim.x + 2*pad0;
    dim_type skStride = shrdLen0 * (blockDim.y + 2*pad1);
    dim_type fStride  = fLen0 * fLen1;
    unsigned batchId  = blockIdx.x/nonBatchBlkSize;
    T *dst            = (T *)out.ptr          + oStep + (batchId*out.strides[3]);
    const T *src      = (const T *)signal.ptr + sStep + (batchId*signal.strides[3]);
    const T *impulse  = (const T *)cFilter;

    dim_type lx = threadIdx.x;
    dim_type ly = threadIdx.y;
    dim_type lz = threadIdx.z;
    dim_type gx = blockDim.x * (blockIdx.x-batchId*nonBatchBlkSize) + lx;
    dim_type gy = blockDim.y * blockIdx.y + ly;
    dim_type gz = blockDim.z * blockIdx.z + lz;
    dim_type i = lx + pad0;
    dim_type j = ly + pad1;
    dim_type k = lz + pad2;

    shrdMem[index(i, j, k, shrdLen0, skStride)] = readSrc(src, gx, gy, gz, signal.dims, signal.strides);

    { //in the hope of limiting scope of l*2 and g*2 variables
        dim_type gx2 = gx + blockDim.x;
        dim_type lx2 = i  + blockDim.x;
        dim_type gy2 = gy + blockDim.y;
        dim_type ly2 = j  + blockDim.y;
        dim_type gz2 = gz + blockDim.z;
        dim_type lz2 = k  + blockDim.z;
        if (lx < pad0) {
            shrdMem[index( lx, j, k, shrdLen0, skStride)] = readSrc(src, gx-pad0, gy, gz, signal.dims, signal.strides);
            shrdMem[index(lx2, j, k, shrdLen0, skStride)] = readSrc(src, gx2    , gy, gz, signal.dims, signal.strides);
        }
        if (ly < pad1) {
            shrdMem[index(i,  ly, k, shrdLen0, skStride)] = readSrc(src, gx, gy-pad1, gz, signal.dims, signal.strides);
            shrdMem[index(i, ly2, k, shrdLen0, skStride)] = readSrc(src, gx, gy2    , gz, signal.dims, signal.strides);
        }
        if (lz < pad2) {
            shrdMem[index(i, j,  lz, shrdLen0, skStride)] = readSrc(src, gx, gy, gz-pad2, signal.dims, signal.strides);
            shrdMem[index(i, j, lz2, shrdLen0, skStride)] = readSrc(src, gx, gy, gz2    , signal.dims, signal.strides);
        }

        if (lx < pad0 && ly < pad1) {
            shrdMem[index( lx,  ly, k, shrdLen0, skStride)] = readSrc(src, gx-pad0, gy-pad1, gz, signal.dims, signal.strides);
            shrdMem[index(lx2,  ly, k, shrdLen0, skStride)] = readSrc(src,     gx2, gy-pad1, gz, signal.dims, signal.strides);
            shrdMem[index( lx, ly2, k, shrdLen0, skStride)] = readSrc(src, gx-pad0,     gy2, gz, signal.dims, signal.strides);
            shrdMem[index(lx2, ly2, k, shrdLen0, skStride)] = readSrc(src,     gx2,     gy2, gz, signal.dims, signal.strides);
        }

        if (ly < pad1 && lz < pad2) {
            shrdMem[index(i,  ly,  lz, shrdLen0, skStride)] = readSrc(src, gx, gy-pad1, gz-pad2, signal.dims, signal.strides);
            shrdMem[index(i, ly2,  lz, shrdLen0, skStride)] = readSrc(src, gx,     gy2, gz-pad2, signal.dims, signal.strides);
            shrdMem[index(i,  ly, lz2, shrdLen0, skStride)] = readSrc(src, gx, gy-pad1,     gz2, signal.dims, signal.strides);
            shrdMem[index(i, ly2, lz2, shrdLen0, skStride)] = readSrc(src, gx,     gy2,     gz2, signal.dims, signal.strides);
        }

        if (lz < pad2 && lx < pad0) {
            shrdMem[index( lx, j,  lz, shrdLen0, skStride)] = readSrc(src, gx-pad0, gy, gz-pad2, signal.dims, signal.strides);
            shrdMem[index(lx2, j,  lz, shrdLen0, skStride)] = readSrc(src,     gx2, gy, gz-pad2, signal.dims, signal.strides);
            shrdMem[index( lx, j, lz2, shrdLen0, skStride)] = readSrc(src, gx-pad0, gy,     gz2, signal.dims, signal.strides);
            shrdMem[index(lx2, j, lz2, shrdLen0, skStride)] = readSrc(src,     gx2, gy,     gz2, signal.dims, signal.strides);
        }

        if (lx < pad0 && ly < pad1 && lz < pad2) {
            shrdMem[index( lx,  ly, lz, shrdLen0, skStride)] = readSrc(src, gx-pad0, gy-pad1, gz-pad2, signal.dims, signal.strides);
            shrdMem[index(lx2,  ly, lz, shrdLen0, skStride)] = readSrc(src, gx2    , gy-pad1, gz-pad2, signal.dims, signal.strides);
            shrdMem[index( lx, ly2, lz, shrdLen0, skStride)] = readSrc(src, gx-pad0, gy2    , gz-pad2, signal.dims, signal.strides);
            shrdMem[index(lx2, ly2, lz, shrdLen0, skStride)] = readSrc(src, gx2    , gy2    , gz-pad2, signal.dims, signal.strides);

            shrdMem[index( lx,  ly, lz2, shrdLen0, skStride)] = readSrc(src, gx-pad0, gy-pad1, gz2, signal.dims, signal.strides);
            shrdMem[index(lx2,  ly, lz2, shrdLen0, skStride)] = readSrc(src, gx2    , gy-pad1, gz2, signal.dims, signal.strides);
            shrdMem[index( lx, ly2, lz2, shrdLen0, skStride)] = readSrc(src, gx-pad0, gy2    , gz2, signal.dims, signal.strides);
            shrdMem[index(lx2, ly2, lz2, shrdLen0, skStride)] = readSrc(src, gx2    , gy2    , gz2, signal.dims, signal.strides);
        }
    }
    __syncthreads();

    if (gx>=0 && gx<out.dims[0] && gy>=0 && gy<out.dims[1] && gz>=0 && gz<out.dims[2]) {
        dim_type ci = i + (expand ? 0 : fLen0/2);
        dim_type cj = j + (expand ? 0 : fLen1/2);
        dim_type ck = k + (expand ? 0 : fLen2/2);

        accType accum = scalar<accType>(0);
        for(dim_type fk=0; fk<fLen2; ++fk) {
            for(dim_type fj=0; fj<fLen1; ++fj) {
                for(dim_type fi=0; fi<fLen0; ++fi) {
                    T f_val = impulse[index(fi, fj, fk, fLen0, fStride)];
                    T s_val = shrdMem[index(ci-fi, cj-fj, ck-fk, shrdLen0, skStride)];
                    accum   = accum + s_val*f_val;
                }
            }
        }
        dst[index(gx, gy, gz, out.strides[1], out.strides[2])] = (T)accum;
    }
}

template<typename T, dim_type baseDim>
void prepareKernelArgs(dim3 &blocks, dim3 &threads, size_t &sharedSize, dim_type &blk_x,
                       ConvolveBatchKind kind, dim_type oDims[], dim_type sDims[], dim_type fDims[])
{
    dim_type blk_y, blk_z;
    if (baseDim==1) {
        threads = dim3(THREADS, 1);
        blk_x   = divup(oDims[0], threads.x);
        blocks  = dim3(blk_x, 1);
        if (kind==MANY2ONE)
            blocks.x *= sDims[1];
        sharedSize = (threads.x+2*(fDims[0]-1)) * sizeof(T);
    } else if (baseDim==2) {
        threads = dim3(THREADS_X, THREADS_Y);
        blk_x   = divup(oDims[0], threads.x);
        blk_y   = divup(oDims[1], threads.y);
        blocks  = dim3(blk_x, blk_y);
        if (kind==MANY2ONE)
            blocks.x *= sDims[2];
        sharedSize = (threads.x+2*(fDims[0]-1))*(threads.y+2*(fDims[1]-1)) * sizeof(T);
    } else if (baseDim==3) {
        threads = dim3(CUBE_X, CUBE_Y, CUBE_Z);
        blk_x   = divup(oDims[0], threads.x);
        blk_y   = divup(oDims[1], threads.y);
        blk_z   = divup(oDims[2], threads.z);
        blocks  = dim3(blk_x, blk_y, blk_z);
        if (kind==MANY2ONE)
            blocks.x *= sDims[3];
        sharedSize = (threads.x+2*(fDims[0]-1)) * (threads.y+2*(fDims[1]-1)) *
                     (threads.z+2*(fDims[2]-1)) * sizeof(T);
    }
}

template<typename T, typename accType, dim_type baseDim, bool expand>
void convolve_nd(Param<T> out, CParam<T> signal, CParam<T> filter, ConvolveBatchKind kind)
{
    dim_type bCount   = 1ll;
    dim_type steps[3] = { 0ll, 0ll, 0ll };
    // [0] - output step, [1] - signal step, [2] - filter step
    if (kind==MANY2MANY) {
        steps[0] = out.strides[baseDim];
        steps[1] = signal.strides[baseDim];
        steps[2] = filter.strides[baseDim];
        bCount   = signal.dims[baseDim];
    } else if (kind==ONE2ALL) {
        steps[0] = out.strides[baseDim];
        steps[2] = filter.strides[baseDim];
        bCount   = filter.dims[baseDim];
    }

    dim3 blocks, threads;
    dim_type blk_x;
    size_t sharedSize;
    prepareKernelArgs<T, baseDim>(blocks, threads, sharedSize, blk_x,
                                  kind, out.dims, signal.dims, filter.dims);

    dim_type filterLen = filter.dims[0];
    for(int i=1; i<baseDim; ++i) filterLen *= filter.dims[i];

    for (dim_type b=0; b<bCount; ++b) {
        // FIX ME: if the filter array is strided, direct copy of symbols
        // might cause issues
        CUDA_CHECK(cudaMemcpyToSymbol(kernel::cFilter, filter.ptr+b*steps[2], filterLen*sizeof(T), 0, cudaMemcpyDeviceToDevice));

        switch(baseDim) {
            case 1:
                (convolve1<T, accType, expand>)
                    <<<blocks, threads, sharedSize>>>(out, signal, filter.dims[0], blk_x, b*steps[0], b*steps[1]);
                break;
            case 2:
                (convolve2<T, accType, expand>)
                    <<<blocks, threads, sharedSize>>>(out, signal, filter.dims[0], filter.dims[1], blk_x,
                                                      b*steps[0], b*steps[1]);
                break;
            case 3:
                (convolve3<T, accType, expand>)
                    <<<blocks, threads, sharedSize>>>(out, signal, filter.dims[0], filter.dims[1], filter.dims[2],
                                                      blk_x, b*steps[0], b*steps[1]);
                break;
        }
    }
    POST_LAUNCH_CHECK();
}

template<typename T>
__device__
T readSrc(T const *src, dim_type i, dim_type j, dim_type dims[], dim_type strides[])
{
    bool is_i = i>=0 && i<dims[0];
    bool is_j = j>=0 && j<dims[1];
    if (is_i && is_j)
        return src[i*strides[0] + j*strides[1]];
    else
        return scalar<T>(0);
}

template<typename T, typename accType, dim_type conv_dim, bool expand>
__global__
void convolve2_separable(Param<T> out, CParam<T> signal, dim_type fLen, dim_type nonBatchBlkSize)
{
    SharedMemory<T> shared;

    T * shrdMem       = shared.getPointer();
    dim_type start    = (expand ? 0 : fLen/2);
    dim_type pad      = fLen-1;
    dim_type shrdLen  = blockDim.x;
    if (conv_dim==0) {
        shrdLen += 2*pad;
    }
    unsigned batchId  = blockIdx.x/nonBatchBlkSize;
    T *dst            = (T *)out.ptr          + (batchId*out.strides[2]);
    const T *src      = (const T *)signal.ptr + (batchId*signal.strides[2]);
    const T *impulse  = (const T *)cFilter;

    dim_type lx = threadIdx.x;
    dim_type ly = threadIdx.y;
    dim_type gx = blockDim.x * (blockIdx.x-batchId*nonBatchBlkSize) + lx;
    dim_type gy = blockDim.y * blockIdx.y + ly;
    dim_type i  = (conv_dim==0 ? lx : ly) + pad;

    if (conv_dim==0) {
        shrdMem[ly*shrdLen+i] = readSrc(src, gx+start, gy+start, signal.dims, signal.strides);
        if (lx < pad) {
            dim_type gx2 = gx + blockDim.x;
            dim_type lx2 = i  + blockDim.x;
            shrdMem[ly*shrdLen+ lx] = readSrc(src, gx-pad+start, gy+start, signal.dims, signal.strides);
            shrdMem[ly*shrdLen+lx2] = readSrc(src, gx2+start   , gy+start, signal.dims, signal.strides);
        }
    } else if (conv_dim==1) {
        shrdMem[i*shrdLen+lx] = readSrc(src, gx+start, gy+start, signal.dims, signal.strides);
        if (ly < pad) {
            dim_type gy2 = gy + blockDim.y;
            dim_type ly2 = i  + blockDim.y;
            shrdMem[ ly*shrdLen+lx] = readSrc(src, gx+start, gy-pad+start, signal.dims, signal.strides);
            shrdMem[ly2*shrdLen+lx] = readSrc(src, gx+start,    gy2+start, signal.dims, signal.strides);
        }
    }
    __syncthreads();

    if (gx>=0 && gx<out.dims[0] && gy>=0 && gy<out.dims[1]) {
        accType accum = scalar<accType>(0);
        for(dim_type f=0; f<fLen; ++f) {
            T f_val = impulse[f];
            dim_type s_idx = (conv_dim==0 ? (ly*shrdLen+(i-f)) : ((i-f)*shrdLen+lx));
            T s_val = shrdMem[s_idx];
            accum   = accum + s_val*f_val;
        }
        dst[gy*out.strides[1]+gx] = (T)accum;
    }
}

template<typename T, typename accType, dim_type conv_dim, bool expand>
void convolve2(Param<T> out, CParam<T> signal, CParam<T> filter)
{
    dim3 threads(THREADS_X, THREADS_Y);

    dim_type blk_x = divup(out.dims[0], threads.x);
    dim_type blk_y = divup(out.dims[1], threads.y);

    dim3 blocks(blk_x*signal.dims[2], blk_y);

    dim_type fLen = filter.dims[0];
    size_t sharedSize = 0;
   if (conv_dim==0)
      sharedSize = (THREADS_X+2*(fLen-1))*THREADS_Y * sizeof(T);
   else if(conv_dim==1)
      sharedSize = (THREADS_Y+2*(fLen-1))*THREADS_X * sizeof(T);

   // FIX ME: if the filter array is strided, direct copy of symbols
   // might cause issues
   CUDA_CHECK(cudaMemcpyToSymbol(kernel::cFilter, filter.ptr,
               fLen*sizeof(T), 0, cudaMemcpyDeviceToDevice));

   (convolve2_separable<T, accType, conv_dim, expand>)
       <<<blocks, threads, sharedSize>>>(out, signal, fLen, blk_x);
}

}

}
