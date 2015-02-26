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
#include <convolve.hpp>

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
static const dim_type MAX_SCONV_FILTER_LEN = 81;

// we shall declare the maximum size required of above all three cases
// and re-use the same constant memory locations for every case
__constant__ char cFilter[2*(2*(MAX_CONV1_FILTER_LEN-1)+THREADS)*sizeof(double)];

__inline__ __device__
dim_type index(dim_type i, dim_type j, dim_type k, dim_type jstride, dim_type kstride)
{
    return i+j*jstride+k*kstride;
}

template<typename T, typename accType, bool expand>
__global__
void convolve1(Param<T> out, CParam<T> signal, dim_type fLen, dim_type nBBS,
               dim_type oStep, dim_type sStep)
{
    SharedMemory<T> shared;
    T * shrdMem = shared.getPointer();

    const dim_type padding = fLen-1;
    const dim_type shrdLen = blockDim.x + 2*padding;
    const unsigned batchId = blockIdx.x/nBBS;

    T *dst           = (T *)out.ptr          + oStep +(batchId*out.strides[1]);
    const T *src     = (const T *)signal.ptr + sStep +(batchId*signal.strides[1]);
    const T *impulse = (const T *)cFilter;

    dim_type gx  = blockDim.x*(blockIdx.x-batchId*nBBS);

    dim_type s0 = signal.strides[0];
    dim_type d0 = signal.dims[0];
    for (dim_type i=threadIdx.x; i<shrdLen; i+=blockDim.x) {
        dim_type idx= gx-padding + i;
        shrdMem[i]  = (idx>=0 && idx<d0) ? src[idx*s0] : scalar<T>(0);
    }
    __syncthreads();
    gx += threadIdx.x;

    if (gx<out.dims[0]) {
        dim_type lx   = threadIdx.x + padding + (expand ? 0 : fLen>>1);
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
               dim_type nBBS, dim_type oStep, dim_type sStep)
{
    SharedMemory<T> shared;
    T * shrdMem       = shared.getPointer();

    const dim_type radius0  = fLen0-1;
    const dim_type radius1  = fLen1-1;
    const dim_type padding0 = 2*radius0;
    const dim_type padding1 = 2*radius1;
    const dim_type shrdLen0 = THREADS_X + padding0;
    const dim_type shrdLen1 = THREADS_Y + padding1;

    unsigned batchId  = blockIdx.x/nBBS;
    T *dst            = (T *)out.ptr          + oStep + (batchId*out.strides[2]);
    const T *src      = (const T *)signal.ptr + sStep + (batchId*signal.strides[2]);
    const T *impulse  = (const T *)cFilter;

    dim_type lx  = threadIdx.x;
    dim_type ly  = threadIdx.y;
    dim_type gx  = THREADS_X * (blockIdx.x-batchId*nBBS) + lx;
    dim_type gy  = THREADS_Y * blockIdx.y + ly;

    dim_type s0 = signal.strides[0];
    dim_type s1 = signal.strides[1];
    dim_type d0 = signal.dims[0];
    dim_type d1 = signal.dims[1];
    // below loops are traditional loops, they only run multiple
    // times filter length is more than launch size
    for (dim_type b=ly, gy2=gy; b<shrdLen1; b+=THREADS_Y, gy2+=THREADS_Y) {
        dim_type j = gy2-radius1;
        bool is_j  = j>=0 && j<d1;
        // move row_set THREADS_Y along coloumns
        for (dim_type a=lx, gx2=gx; a<shrdLen0; a+=THREADS_X, gx2+=THREADS_X) {
            dim_type i = gx2-radius0;
            bool is_i  = i>=0 && i<d0;
            shrdMem[b*shrdLen0+a] = (is_i && is_j ? src[i*s0+j*s1] : scalar<T>(0));
        }
    }
    __syncthreads();

    if (gx<out.dims[0] && gy<out.dims[1]) {
        dim_type ci = lx + radius0 + (expand ? 0 : fLen0>>1);
        dim_type cj = ly + radius1 + (expand ? 0 : fLen1>>1);

        accType accum = scalar<accType>(0);
#pragma unroll
        for(dim_type fj=0; fj<fLen1; ++fj) {
#pragma unroll
            for(dim_type fi=0; fi<fLen0; ++fi) {
                T f_val = impulse[fj*fLen0+fi];
                T s_val = shrdMem[(cj-fj)*shrdLen0 + (ci-fi)];
                accum   = accum + s_val*f_val;
            }
        }
        dst[gy*out.strides[1]+gx] = (T)accum;
    }
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
void convolve3(Param<T> out, CParam<T> signal, dim_type fLen0, dim_type fLen1,
               dim_type fLen2, dim_type nBBS, dim_type oStep, dim_type sStep)
{
    SharedMemory<T> shared;

    T * shrdMem       = shared.getPointer();
    dim_type radius0  = fLen0-1;
    dim_type radius1  = fLen1-1;
    dim_type radius2  = fLen2-1;
    dim_type padding0 = 2*radius0;
    dim_type padding1 = 2*radius1;
    dim_type padding2 = 2*radius2;
    dim_type shrdLen0 = blockDim.x + padding0;
    dim_type skStride = shrdLen0 * (blockDim.y + padding1);
    dim_type fStride  = fLen0 * fLen1;
    unsigned batchId  = blockIdx.x/nBBS;

    T *dst            = (T *)out.ptr          + oStep + (batchId*out.strides[3]);
    const T *src      = (const T *)signal.ptr + sStep + (batchId*signal.strides[3]);
    const T *impulse  = (const T *)cFilter;

    dim_type lx  = threadIdx.x;
    dim_type ly  = threadIdx.y;
    dim_type lz  = threadIdx.z;
    dim_type gx  = blockDim.x * (blockIdx.x-batchId*nBBS) + lx;
    dim_type gy  = blockDim.y * blockIdx.y + ly;
    dim_type gz  = blockDim.z * blockIdx.z + lz;
    dim_type lx2 = lx  + blockDim.x;
    dim_type ly2 = ly  + blockDim.y;
    dim_type lz2 = lz  + blockDim.z;
    dim_type gx2 = gx + blockDim.x;
    dim_type gy2 = gy + blockDim.y;
    dim_type gz2 = gz + blockDim.z;

    shrdMem[index(lx, ly, lz, shrdLen0, skStride)] =
        readSrc(src, gx-radius0, gy-radius1, gz-radius2, signal.dims, signal.strides);

    if (lx < padding0) {
        shrdMem[index(lx2, ly, lz, shrdLen0, skStride)] =
            readSrc(src, gx2-radius0, gy-radius1, gz-radius2, signal.dims, signal.strides);
    }
    if (ly < padding1) {
        shrdMem[index(lx, ly2, lz, shrdLen0, skStride)] =
            readSrc(src, gx-radius0, gy2-radius1, gz-radius2, signal.dims, signal.strides);
    }
    if (lz < padding2) {
        shrdMem[index(lx, ly, lz2, shrdLen0, skStride)] =
            readSrc(src, gx-radius0, gy-radius1, gz2-radius2, signal.dims, signal.strides);
    }

    if (lx < padding0 && ly < padding1) {
        shrdMem[index(lx2, ly2, lz, shrdLen0, skStride)] =
            readSrc(src, gx2-radius0, gy2-radius1, gz-radius2, signal.dims, signal.strides);
    }

    if (ly < padding1 && lz < padding2) {
        shrdMem[index(lx, ly2, lz2, shrdLen0, skStride)] =
            readSrc(src, gx-radius0, gy2-radius1, gz2-radius2, signal.dims, signal.strides);
    }

    if (lz < padding2 && lx < padding0) {
        shrdMem[index(lx2, ly, lz2, shrdLen0, skStride)] =
            readSrc(src, gx2-radius0, gy-radius1, gz2-radius2, signal.dims, signal.strides);
    }

    if (lx < padding0 && ly < padding1 && lz < padding2) {
        shrdMem[index(lx2, ly2, lz2, shrdLen0, skStride)] =
            readSrc(src, gx2-radius0, gy2-radius1, gz2-radius2, signal.dims, signal.strides);
    }

    __syncthreads();

    if (gx<out.dims[0] && gy<out.dims[1] && gz<out.dims[2]) {
        dim_type ci = lx + radius0 + (expand ? 0 : fLen0>>1);
        dim_type cj = ly + radius1 + (expand ? 0 : fLen1>>1);
        dim_type ck = lz + radius2 + (expand ? 0 : fLen2>>1);

        accType accum = scalar<accType>(0);
#pragma unroll
        for(dim_type fk=0; fk<fLen2; ++fk) {
#pragma unroll
            for(dim_type fj=0; fj<fLen1; ++fj) {
#pragma unroll
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
    bool callKernel = true;

    dim_type MCFL2 = kernel::MAX_CONV2_FILTER_LEN;
    dim_type MCFL3 = kernel::MAX_CONV3_FILTER_LEN;
    switch(baseDim) {
        case 1:
            if (filter.dims[0]>kernel::MAX_CONV1_FILTER_LEN)
                callKernel = false;
            break;
        case 2:
            if ((filter.dims[0]*filter.dims[1]) > (MCFL2 * MCFL2))
                callKernel = false;
            break;
        case 3:
            if ((filter.dims[0]*filter.dims[1]*filter.dims[2]) > (MCFL3 * MCFL3 * MCFL3))
                callKernel = false;
            break;
    }

    if (!callKernel) {
        CUDA_NOT_SUPPORTED();
    }

    dim_type bCount   = 1;
    dim_type steps[3] = { 0, 0, 0 };
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
        // FIXME: if the filter array is strided, direct copy of symbols
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

template<typename T, typename accType, dim_type conv_dim, bool expand>
__global__
void convolve2_separable(Param<T> out, CParam<T> signal, dim_type fLen, dim_type nBBS)
{
    SharedMemory<T> shared;
    T * shrdMem = shared.getPointer();

    const dim_type radius  = fLen-1;
    const dim_type padding = 2*radius;
    const dim_type s0      = signal.strides[0];
    const dim_type s1      = signal.strides[1];
    const dim_type d0      = signal.dims[0];
    const dim_type d1      = signal.dims[1];

    dim_type shrdLen = THREADS_X;
    if (conv_dim==0) {
        shrdLen += padding;
    }

    unsigned batchId  = blockIdx.x/nBBS;
    T *dst            = (T *)out.ptr          + (batchId*out.strides[2]);
    const T *src      = (const T *)signal.ptr + (batchId*signal.strides[2]);
    const T *impulse  = (const T *)cFilter;

    dim_type lx = threadIdx.x;
    dim_type ly = threadIdx.y;
    dim_type ox = THREADS_X * (blockIdx.x-batchId*nBBS) + lx;
    dim_type oy = THREADS_Y * blockIdx.y + ly;
    dim_type gx = ox;
    dim_type gy = oy;

    // below if-else statement is based on template parameter
    if (conv_dim==0) {
        gx += (expand ? 0 : fLen>>1);
        dim_type endX = ((fLen-1)<<1) + THREADS_X;

        for(dim_type lx = threadIdx.x, glb_x = gx; lx<endX; lx += THREADS_X, glb_x += THREADS_X) {
            dim_type i = glb_x - radius;
            dim_type j = gy;
            bool is_i  = i>=0 && i<d0;
            bool is_j  = j>=0 && j<d1;
            shrdMem[ly*shrdLen+lx] = (is_i && is_j ? src[i*s0 + j*s1] : scalar<T>(0));
        }

    } else if (conv_dim==1) {
        gy += (expand ? 0 : fLen>>1);
        dim_type endY = ((fLen-1)<<1) + THREADS_Y;

        for(dim_type ly = threadIdx.y, glb_y = gy; ly<endY; ly += THREADS_Y, glb_y += THREADS_Y) {
            dim_type i = gx;
            dim_type j = glb_y - radius;
            bool is_i  = i>=0 && i<d0;
            bool is_j  = j>=0 && j<d1;
            shrdMem[ly*shrdLen+lx] = (is_i && is_j ? src[i*s0 + j*s1] : scalar<T>(0));
        }
    }
    __syncthreads();

    if (ox<out.dims[0] && oy<out.dims[1]) {
        // below conditional statement is based on template parameter
        dim_type i  = (conv_dim==0 ? lx : ly) + radius;
        accType accum = scalar<accType>(0);
        for(dim_type f=0; f<fLen; ++f) {
            T f_val = impulse[f];
            // below conditional statement is based on template parameter
            dim_type s_idx = (conv_dim==0 ? (ly*shrdLen+(i-f)) : ((i-f)*shrdLen+lx));
            T s_val = shrdMem[s_idx];
            accum   = accum + s_val*f_val;
        }
        dst[oy*out.strides[1]+ox] = (T)accum;
    }
}

template<typename T, typename accType, dim_type conv_dim, bool expand>
void convolve2(Param<T> out, CParam<T> signal, CParam<T> filter)
{
   if(filter.dims[0] > kernel::MAX_SCONV_FILTER_LEN) {
        // call upon fft
        CUDA_NOT_SUPPORTED();
    }

    dim3 threads(THREADS_X, THREADS_Y);

    dim_type blk_x = divup(out.dims[0], threads.x);
    dim_type blk_y = divup(out.dims[1], threads.y);

    dim3 blocks(blk_x*signal.dims[2], blk_y);

    dim_type fLen = filter.dims[0];
    size_t sharedSize = 0;
   if (conv_dim==0)
      sharedSize = (threads.x+2*(fLen-1))*threads.y * sizeof(T);
   else if(conv_dim==1)
      sharedSize = (threads.y+2*(fLen-1))*threads.x * sizeof(T);

   // FIX ME: if the filter array is strided, direct copy of symbols
   // might cause issues
   CUDA_CHECK(cudaMemcpyToSymbol(kernel::cFilter, filter.ptr,
               fLen*sizeof(T), 0, cudaMemcpyDeviceToDevice));

   (convolve2_separable<T, accType, conv_dim, expand>)
       <<<blocks, threads, sharedSize>>>(out, signal, fLen, blk_x);

   POST_LAUNCH_CHECK();
}

#define INSTANTIATE(T, accT)  \
	template void convolve_nd<T, accT, 1, true>(Param<T> out, CParam<T> signal, CParam<T> filter, ConvolveBatchKind kind);\
	template void convolve_nd<T, accT, 1, false>(Param<T> out, CParam<T> signal, CParam<T> filter, ConvolveBatchKind kind);\
	template void convolve_nd<T, accT, 2, true>(Param<T> out, CParam<T> signal, CParam<T> filter, ConvolveBatchKind kind);\
	template void convolve_nd<T, accT, 2, false>(Param<T> out, CParam<T> signal, CParam<T> filter, ConvolveBatchKind kind);\
	template void convolve_nd<T, accT, 3, true>(Param<T> out, CParam<T> signal, CParam<T> filter, ConvolveBatchKind kind);\
	template void convolve_nd<T, accT, 3, false>(Param<T> out, CParam<T> signal, CParam<T> filter, ConvolveBatchKind kind);\
	template void convolve2<T, accT, 0, true>(Param<T> out, CParam<T> signal, CParam<T> filter);\
	template void convolve2<T, accT, 0, false>(Param<T> out, CParam<T> signal, CParam<T> filter);\
	template void convolve2<T, accT, 1, true>(Param<T> out, CParam<T> signal, CParam<T> filter);\
	template void convolve2<T, accT, 1, false>(Param<T> out, CParam<T> signal, CParam<T> filter);\


INSTANTIATE(cdouble, cdouble)
INSTANTIATE(cfloat ,  cfloat)
INSTANTIATE(double ,  double)
INSTANTIATE(float  ,   float)
INSTANTIATE(uint   ,   float)
INSTANTIATE(int    ,   float)
INSTANTIATE(uchar  ,   float)
INSTANTIATE(char   ,   float)

}

}
