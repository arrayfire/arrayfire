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

static const int THREADS   = 256;

static const int THREADS_X = 16;
static const int THREADS_Y = 16;

static const int CUBE_X    =  8;
static const int CUBE_Y    =  8;
static const int CUBE_Z    =  4;

// below shared MAX_*_LEN's are calculated based on
// a maximum shared memory configuration of 48KB per block
// considering complex types as well
static const int MAX_CONV1_FILTER_LEN = 129;
static const int MAX_CONV2_FILTER_LEN = 17;
static const int MAX_CONV3_FILTER_LEN = 5;

// we shall declare the maximum size required of above all three cases
// and re-use the same constant memory locations for every case
__constant__ char cFilter[2*(2*(MAX_CONV1_FILTER_LEN-1)+THREADS)*sizeof(double)];

template<typename T, typename aT, bool expand>
__global__
void convolve1(Param<T> out, CParam<T> signal, int fLen,
               int nBBS0, int nBBS1,
               int o1, int o2, int o3,
               int s1, int s2, int s3)
{
    SharedMemory<T> shared;
    T * shrdMem = shared.getPointer();

    const int padding = fLen-1;
    const int shrdLen = blockDim.x + 2*padding;
    const unsigned b1 = blockIdx.x/nBBS0;   /* [0 {1} 2 3] */
    const unsigned b3 = blockIdx.y/nBBS1;   /* [0 1 2 {3}] */
    const unsigned b2 = blockIdx.y-nBBS1*b3;/* [0 1 {2} 3] */

    T *dst = (T *)out.ptr + (b1 * out.strides[1] +  /* activated with batched input signal */
                             o1 * out.strides[1] +  /* activated with batched input filter */
                             b2 * out.strides[2] +  /* activated with batched input signal */
                             o2 * out.strides[2] +  /* activated with batched input filter */
                             b3 * out.strides[3] +  /* activated with batched input signal */
                             o3 * out.strides[3]);  /* activated with batched input filter */

    const T *src = (const T *)signal.ptr + (b1 * signal.strides[1] + /* activated with batched input signal */
                                            s1 * signal.strides[1] + /* activated with batched input filter */
                                            b2 * signal.strides[2] + /* activated with batched input signal */
                                            s2 * signal.strides[2] + /* activated with batched input filter */
                                            b3 * signal.strides[3] + /* activated with batched input signal */
                                            s3 * signal.strides[3]); /* activated with batched input filter */

    const aT *impulse = (const aT *)cFilter;

    int gx  = blockDim.x*(blockIdx.x-b1*nBBS0);

    int s0 = signal.strides[0];
    int d0 = signal.dims[0];
    for (int i=threadIdx.x; i<shrdLen; i+=blockDim.x) {
        int idx= gx-padding + i;
        shrdMem[i]  = (idx>=0 && idx<d0) ? src[idx*s0] : scalar<T>(0);
    }
    __syncthreads();
    gx += threadIdx.x;

    if (gx<out.dims[0]) {
        int lx   = threadIdx.x + padding + (expand ? 0 : fLen>>1);
        aT accum = scalar<aT>(0);
        for(int f=0; f<fLen; ++f) {
            accum = accum + (shrdMem[lx-f]*impulse[f]);
        }
        dst[gx] = (T)accum;
    }
}

template<typename T, typename aT, bool expand, int fLen0, int fLen1>
__global__
void convolve2(Param<T> out, CParam<T> signal, int nBBS0,
               int nBBS1, int o2, int o3, int s2, int s3)
{
    const size_t C_SIZE  = (THREADS_X+2*(fLen0-1))* (THREADS_Y+2*(fLen1-1));
    __shared__ T shrdMem[C_SIZE];

    const int radius0  = fLen0-1;
    const int radius1  = fLen1-1;
    const int padding0 = 2*radius0;
    const int padding1 = 2*radius1;
    const int shrdLen0 = THREADS_X + padding0;
    const int shrdLen1 = THREADS_Y + padding1;

    unsigned b0  = blockIdx.x/nBBS0;
    unsigned b1  = blockIdx.y/nBBS1;
    T *dst = (T *)out.ptr + (b0 * out.strides[2] + /* activated with batched input signal */
                             o2 * out.strides[2] + /* activated with batched input filter */
                             b1 * out.strides[3] + /* activated with batched input signal */
                             o3 * out.strides[3]); /* activated with batched input filter */

    const T *src = (const T *)signal.ptr + (b0 * signal.strides[2] + /* activated with batched input signal */
                                            s2 * signal.strides[2] + /* activated with batched input filter */
                                            b1 * signal.strides[3] + /* activated with batched input signal */
                                            s3 * signal.strides[3]); /* activated with batched input filter */

    const aT *impulse  = (const aT *)cFilter;

    int lx  = threadIdx.x;
    int ly  = threadIdx.y;
    int gx  = THREADS_X * (blockIdx.x-b0*nBBS0) + lx;
    int gy  = THREADS_Y * (blockIdx.y-b1*nBBS1) + ly;

    int s0 = signal.strides[0];
    int s1 = signal.strides[1];
    int d0 = signal.dims[0];
    int d1 = signal.dims[1];
    // below loops are traditional loops, they only run multiple
    // times filter length is more than launch size
#pragma unroll
    for (int b=ly, gy2=gy; b<shrdLen1; b+=THREADS_Y, gy2+=THREADS_Y) {
        int j = gy2-radius1;
        bool is_j  = j>=0 && j<d1;
        // move row_set THREADS_Y along coloumns
#pragma unroll
        for (int a=lx, gx2=gx; a<shrdLen0; a+=THREADS_X, gx2+=THREADS_X) {
            int i = gx2-radius0;
            bool is_i  = i>=0 && i<d0;
            shrdMem[b*shrdLen0+a] = (is_i && is_j ? src[i*s0+j*s1] : scalar<T>(0));
        }
    }
    __syncthreads();

    if (gx<out.dims[0] && gy<out.dims[1]) {
        int ci = lx + radius0 + (expand ? 0 : fLen0>>1);
        int cj = ly + radius1 + (expand ? 0 : fLen1>>1);

        aT accum = scalar<aT>(0);
#pragma unroll
        for(int fj=0; fj<fLen1; ++fj) {
#pragma unroll
            for(int fi=0; fi<fLen0; ++fi) {
                aT f_val = impulse[fj*fLen0+fi];
                T s_val = shrdMem[(cj-fj)*shrdLen0 + (ci-fi)];
                accum   = accum + s_val*f_val;
            }
        }
        dst[gy*out.strides[1]+gx] = (T)accum;
    }
}

__inline__ __device__
int index(int i, int j, int k, int jstride, int kstride)
{
    return i+j*jstride+k*kstride;
}

template<typename T, typename aT, bool expand>
__global__
void convolve3(Param<T> out, CParam<T> signal, int fLen0, int fLen1,
               int fLen2, int nBBS, int o3, int s3)
{
    SharedMemory<T> shared;

    T * shrdMem       = shared.getPointer();
    int radius0  = fLen0-1;
    int radius1  = fLen1-1;
    int radius2  = fLen2-1;
    int shrdLen0 = blockDim.x + 2*radius0;
    int shrdLen1 = blockDim.y + 2*radius1;
    int shrdLen2 = blockDim.z + 2*radius2;
    int skStride = shrdLen0 * shrdLen1;
    int fStride  = fLen0 * fLen1;
    unsigned b2  = blockIdx.x/nBBS;

    T *dst = (T *)out.ptr + (b2 * out.strides[3] + /* activated with batched input signal */
                             o3 * out.strides[3]); /* activated with batched input filter */

    const T *src = (const T *)signal.ptr + (b2 * signal.strides[3] + /* activated with batched input signal */
                                            s3 * signal.strides[3]); /* activated with batched input filter */

    const aT *impulse  = (const aT *)cFilter;

    int lx  = threadIdx.x;
    int ly  = threadIdx.y;
    int lz  = threadIdx.z;
    int gx  = blockDim.x * (blockIdx.x-b2*nBBS) + lx;
    int gy  = blockDim.y * blockIdx.y + ly;
    int gz  = blockDim.z * blockIdx.z + lz;

    int s0 = signal.strides[0];
    int s1 = signal.strides[1];
    int s2 = signal.strides[2];
    int d0 = signal.dims[0];
    int d1 = signal.dims[1];
    int d2 = signal.dims[2];
#pragma unroll
    for (int c=lz, gz2=gz; c<shrdLen2; c+=CUBE_Z, gz2+=CUBE_Z) {
        int k = gz2-radius2;
        bool is_k  = k>=0 && k<d2;
#pragma unroll
        for (int b=ly, gy2=gy; b<shrdLen1; b+=CUBE_Y, gy2+=CUBE_Y) {
            int j = gy2-radius1;
            bool is_j  = j>=0 && j<d1;
#pragma unroll
            for (int a=lx, gx2=gx; a<shrdLen0; a+=CUBE_X, gx2+=CUBE_X) {
                int i = gx2-radius0;
                bool is_i  = i>=0 && i<d0;
                shrdMem[c*skStride+b*shrdLen0+a] =
                    (is_i && is_j && is_k ? src[i*s0+j*s1+k*s2] : scalar<T>(0));
            }
        }
    }
    __syncthreads();

    if (gx<out.dims[0] && gy<out.dims[1] && gz<out.dims[2]) {
        int ci = lx + radius0 + (expand ? 0 : fLen0>>1);
        int cj = ly + radius1 + (expand ? 0 : fLen1>>1);
        int ck = lz + radius2 + (expand ? 0 : fLen2>>1);

        aT accum = scalar<aT>(0);
#pragma unroll
        for(int fk=0; fk<fLen2; ++fk) {
#pragma unroll
            for(int fj=0; fj<fLen1; ++fj) {
#pragma unroll
                for(int fi=0; fi<fLen0; ++fi) {
                    aT f_val = impulse[index(fi, fj, fk, fLen0, fStride)];
                    T s_val = shrdMem[index(ci-fi, cj-fj, ck-fk, shrdLen0, skStride)];
                    accum   = accum + s_val*f_val;
                }
            }
        }
        dst[index(gx, gy, gz, out.strides[1], out.strides[2])] = (T)accum;
    }
}

struct conv_kparam_t {
    dim3              mBlocks;
    dim3             mThreads;
    size_t        mSharedSize;
    int           mBlk_x;
    int           mBlk_y;
    bool       outHasNoOffset;
    bool        inHasNoOffset;
    bool     launchMoreBlocks;
    int             o[3];
    int             s[3];
};

template<typename T>
void prepareKernelArgs(conv_kparam_t &params, dim_t oDims[], dim_t fDims[], int baseDim)
{
    int batchDims[4] = {1, 1, 1, 1};
    for(int i=baseDim; i<4; ++i) {
        batchDims[i] = (params.launchMoreBlocks ? 1 : oDims[i]);
    }

    if (baseDim==1) {
        params.mThreads    = dim3(THREADS, 1);
        params.mBlk_x      = divup(oDims[0], params.mThreads.x);
        params.mBlk_y      = batchDims[2];
        params.mBlocks     = dim3(params.mBlk_x * batchDims[1], params.mBlk_y * batchDims[3]);
        params.mSharedSize = (params.mThreads.x+2*(fDims[0]-1)) * sizeof(T);
    } else if (baseDim==2) {
        params.mThreads    = dim3(THREADS_X, THREADS_Y);
        params.mBlk_x      = divup(oDims[0], params.mThreads.x);
        params.mBlk_y      = divup(oDims[1], params.mThreads.y);
        params.mBlocks     = dim3(params.mBlk_x * batchDims[2], params.mBlk_y * batchDims[3]);
    } else if (baseDim==3) {
        params.mThreads    = dim3(CUBE_X, CUBE_Y, CUBE_Z);
        params.mBlk_x      = divup(oDims[0], params.mThreads.x);
        params.mBlk_y      = divup(oDims[1], params.mThreads.y);
        int blk_z     = divup(oDims[2], params.mThreads.z);
        params.mBlocks     = dim3(params.mBlk_x * batchDims[3], params.mBlk_y, blk_z);
        params.mSharedSize = (params.mThreads.x+2*(fDims[0]-1)) *
                             (params.mThreads.y+2*(fDims[1]-1)) *
                             (params.mThreads.z+2*(fDims[2]-1)) * sizeof(T);
    }
}

template<typename T, typename aT, bool expand, int f0, int f1>
void conv2Helper(const conv_kparam_t &p, Param<T> out, CParam<T> sig)
{
    (convolve2<T, aT, expand, f0, f1>)
        <<<p.mBlocks, p.mThreads>>>(out, sig, p.mBlk_x, p.mBlk_y, p.o[1], p.o[2], p.s[1], p.s[2]);

    POST_LAUNCH_CHECK();
}

template<typename T, typename aT, bool expand, int f0>
void conv2Helper(const conv_kparam_t &p, Param<T> out, CParam<T> sig, int f1)
{
    switch(f1) {
        case  1: conv2Helper<T, aT, expand, f0,  1>(p, out, sig); break;
        case  2: conv2Helper<T, aT, expand, f0,  2>(p, out, sig); break;
        case  3: conv2Helper<T, aT, expand, f0,  3>(p, out, sig); break;
        case  4: conv2Helper<T, aT, expand, f0,  4>(p, out, sig); break;
        case  5: conv2Helper<T, aT, expand, f0,  5>(p, out, sig); break;
        default: CUDA_NOT_SUPPORTED();
    }
}

template<typename T, typename aT, bool expand>
void conv2Helper(const conv_kparam_t &p, Param<T> out, CParam<T> sig, int f0, int f1)
{
    switch(f0) {
        case  1: conv2Helper<T, aT, expand,  1>(p, out, sig, f1); break;
        case  2: conv2Helper<T, aT, expand,  2>(p, out, sig, f1); break;
        case  3: conv2Helper<T, aT, expand,  3>(p, out, sig, f1); break;
        case  4: conv2Helper<T, aT, expand,  4>(p, out, sig, f1); break;
        case  5: conv2Helper<T, aT, expand,  5>(p, out, sig, f1); break;
        default: {
                     if (f0==f1) {
                         switch(f1) {
                             case  6: conv2Helper<T, aT, expand,  6,  6>(p, out, sig); break;
                             case  7: conv2Helper<T, aT, expand,  7,  7>(p, out, sig); break;
                             case  8: conv2Helper<T, aT, expand,  8,  8>(p, out, sig); break;
                             case  9: conv2Helper<T, aT, expand,  9,  9>(p, out, sig); break;
                             case 10: conv2Helper<T, aT, expand, 10, 10>(p, out, sig); break;
                             case 11: conv2Helper<T, aT, expand, 11, 11>(p, out, sig); break;
                             case 12: conv2Helper<T, aT, expand, 12, 12>(p, out, sig); break;
                             case 13: conv2Helper<T, aT, expand, 13, 13>(p, out, sig); break;
                             case 14: conv2Helper<T, aT, expand, 14, 14>(p, out, sig); break;
                             case 15: conv2Helper<T, aT, expand, 15, 15>(p, out, sig); break;
                             case 16: conv2Helper<T, aT, expand, 16, 16>(p, out, sig); break;
                             case 17: conv2Helper<T, aT, expand, 17, 17>(p, out, sig); break;
                             default: CUDA_NOT_SUPPORTED();
                         }
                     } else
                         CUDA_NOT_SUPPORTED();
                 } break;
    }
}

template<typename T, typename aT, bool expand>
void convolve_1d(conv_kparam_t &p, Param<T> out, CParam<T> sig, CParam<aT> filt)
{
    prepareKernelArgs<T>(p, out.dims, filt.dims, 1);

    int filterLen = filt.dims[0];

    for (int b3=0; b3<filt.dims[3]; ++b3) {
        int f3Off = b3 * filt.strides[3];

        for (int b2=0; b2<filt.dims[2]; ++b2) {
            int f2Off = b2 * filt.strides[2];

            for (int b1=0; b1<filt.dims[1]; ++b1) {
                int f1Off = b1 * filt.strides[1];

                // FIXME: if the filter array is strided, direct copy of symbols
                // might cause issues
                CUDA_CHECK(cudaMemcpyToSymbol(kernel::cFilter,
                                              filt.ptr+(f1Off+f2Off+f3Off),
                                              filterLen*sizeof(aT),
                                              0, cudaMemcpyDeviceToDevice));

                p.o[0] = (p.outHasNoOffset ? 0 : b1);
                p.o[1] = (p.outHasNoOffset ? 0 : b2);
                p.o[2] = (p.outHasNoOffset ? 0 : b3);
                p.s[0] = (p.inHasNoOffset ? 0 : b1);
                p.s[1] = (p.inHasNoOffset ? 0 : b2);
                p.s[2] = (p.inHasNoOffset ? 0 : b3);

                (convolve1<T, aT, expand>)
                    <<<p.mBlocks, p.mThreads, p.mSharedSize>>>
                    (out, sig, filt.dims[0], p.mBlk_x, p.mBlk_y,
                     p.o[0], p.o[1], p.o[2], p.s[0], p.s[1], p.s[2]);

                POST_LAUNCH_CHECK();
            }
        }
    }
}

template<typename T, typename aT, bool expand>
void convolve_2d(conv_kparam_t &p, Param<T> out, CParam<T> sig, CParam<aT> filt)
{
    prepareKernelArgs<T>(p, out.dims, filt.dims, 2);

    int filterLen = filt.dims[0] * filt.dims[1];

    for (int b3=0; b3<filt.dims[3]; ++b3) {
        int f3Off = b3 * filt.strides[3];

        for (int b2=0; b2<filt.dims[2]; ++b2) {
            int f2Off = b2 * filt.strides[2];

            // FIXME: if the filter array is strided, direct copy of symbols
            // might cause issues
            CUDA_CHECK(cudaMemcpyToSymbol(kernel::cFilter,
                                          filt.ptr+(f2Off+f3Off),
                                          filterLen*sizeof(aT),
                                          0, cudaMemcpyDeviceToDevice));

            p.o[1] = (p.outHasNoOffset ? 0 : b2);
            p.o[2] = (p.outHasNoOffset ? 0 : b3);
            p.s[1] = (p.inHasNoOffset ? 0 : b2);
            p.s[2] = (p.inHasNoOffset ? 0 : b3);

            conv2Helper<T, aT, expand>(p, out, sig, filt.dims[0], filt.dims[1]);
        }
    }
}

template<typename T, typename aT, bool expand>
void convolve_3d(conv_kparam_t &p, Param<T> out, CParam<T> sig, CParam<aT> filt)
{
    prepareKernelArgs<T>(p, out.dims, filt.dims, 3);

    int filterLen = filt.dims[0] * filt.dims[1] * filt.dims[2];

    for (int b3=0; b3<filt.dims[3]; ++b3) {
        int f3Off = b3 * filt.strides[3];

        // FIXME: if the filter array is strided, direct copy of symbols
        // might cause issues
        CUDA_CHECK(cudaMemcpyToSymbol(kernel::cFilter,
                    filt.ptr+f3Off,
                    filterLen*sizeof(aT),
                    0, cudaMemcpyDeviceToDevice));

        p.o[2] = (p.outHasNoOffset ? 0 : b3);
        p.s[2] = (p.inHasNoOffset ? 0 : b3);

        (convolve3<T, aT, expand>)
            <<<p.mBlocks, p.mThreads, p.mSharedSize>>>
            (out, sig, filt.dims[0], filt.dims[1], filt.dims[2], p.mBlk_x, p.o[2], p.s[2]);

        POST_LAUNCH_CHECK();
    }
}

template<typename T, typename aT, int baseDim, bool expand>
void convolve_nd(Param<T> out, CParam<T> signal, CParam<aT> filt, ConvolveBatchKind kind)
{
    bool callKernel = true;

    int MCFL2 = kernel::MAX_CONV2_FILTER_LEN;
    int MCFL3 = kernel::MAX_CONV3_FILTER_LEN;
    switch(baseDim) {
        case 1: if (filt.dims[0]>kernel::MAX_CONV1_FILTER_LEN) callKernel = false; break;
        case 2: if ((filt.dims[0]*filt.dims[1]) > (MCFL2 * MCFL2)) callKernel = false; break;
        case 3: if ((filt.dims[0]*filt.dims[1]*filt.dims[2]) > (MCFL3 * MCFL3 * MCFL3)) callKernel = false; break;
    }

    if (!callKernel) { CUDA_NOT_SUPPORTED(); }

    conv_kparam_t param;
    for (int i=0; i<3; ++i) {
        param.o[i] = 0;
        param.s[i] = 0;
    }
    param.launchMoreBlocks = kind==CONVOLVE_BATCH_SAME || kind==CONVOLVE_BATCH_KERNEL;
    param.outHasNoOffset = kind==CONVOLVE_BATCH_SIGNAL || kind==CONVOLVE_BATCH_NONE;
    param.inHasNoOffset  = kind!=CONVOLVE_BATCH_SAME;

    switch(baseDim) {
        case 1: convolve_1d<T, aT, expand>(param, out, signal, filt); break;
        case 2: convolve_2d<T, aT, expand>(param, out, signal, filt); break;
        case 3: convolve_3d<T, aT, expand>(param, out, signal, filt); break;
    }

    POST_LAUNCH_CHECK();
}

#define INSTANTIATE(T, aT)  \
	template void convolve_nd<T, aT, 1, true >(Param<T> out, CParam<T> signal, CParam<aT> filter, ConvolveBatchKind kind);\
	template void convolve_nd<T, aT, 1, false>(Param<T> out, CParam<T> signal, CParam<aT> filter, ConvolveBatchKind kind);\
	template void convolve_nd<T, aT, 2, true >(Param<T> out, CParam<T> signal, CParam<aT> filter, ConvolveBatchKind kind);\
	template void convolve_nd<T, aT, 2, false>(Param<T> out, CParam<T> signal, CParam<aT> filter, ConvolveBatchKind kind);\
	template void convolve_nd<T, aT, 3, true >(Param<T> out, CParam<T> signal, CParam<aT> filter, ConvolveBatchKind kind);\
	template void convolve_nd<T, aT, 3, false>(Param<T> out, CParam<T> signal, CParam<aT> filter, ConvolveBatchKind kind);\


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
