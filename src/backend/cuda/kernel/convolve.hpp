/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <common/defines.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_cuda.hpp>
#include <nvrtc_kernel_headers/convolve1_cuh.hpp>
#include <nvrtc_kernel_headers/convolve2_cuh.hpp>
#include <nvrtc_kernel_headers/convolve3_cuh.hpp>
#include <nvrtc_kernel_headers/convolve_separable_cuh.hpp>
#include <traits.hpp>

namespace arrayfire {
namespace cuda {
namespace kernel {

static const int CONV_THREADS = 256;

static const int CONV2_THREADS_X = 16;
static const int CONV2_THREADS_Y = 16;

static const int CONV3_CUBE_X = 8;
static const int CONV3_CUBE_Y = 8;
static const int CONV3_CUBE_Z = 4;

// below shared MAX_*_LEN's are calculated based on
// a maximum shared memory configuration of 48KB per block
// considering complex types as well
static const int MAX_CONV1_FILTER_LEN = 129;
static const int MAX_CONV2_FILTER_LEN = 17;
static const int MAX_CONV3_FILTER_LEN = 5;

constexpr static const char* conv_c_name  = "cFilter";
constexpr static const char* sconv_c_name = "sFilter";

struct conv_kparam_t {
    dim3 mBlocks;
    dim3 mThreads;
    size_t mSharedSize;
    int mBlk_x;
    int mBlk_y;
    bool outHasNoOffset;
    bool inHasNoOffset;
    bool launchMoreBlocks;
    int o[3];
    int s[3];
};

template<typename T>
void prepareKernelArgs(conv_kparam_t& params, dim_t oDims[], dim_t fDims[],
                       int baseDim) {
    int batchDims[4] = {1, 1, 1, 1};
    for (int i = baseDim; i < 4; ++i) {
        batchDims[i] = (params.launchMoreBlocks ? 1 : oDims[i]);
    }

    const int maxBlocksY = getDeviceProp(getActiveDeviceId()).maxGridSize[1];
    if (baseDim == 1) {
        params.mThreads = dim3(CONV_THREADS, 1);
        params.mBlk_x   = divup(oDims[0], params.mThreads.x);
        params.mBlk_y   = batchDims[2];
        params.mBlocks =
            dim3(params.mBlk_x * batchDims[1], params.mBlk_y * batchDims[3]);
        params.mSharedSize =
            (params.mThreads.x + 2 * (fDims[0] - 1)) * sizeof(T);
        params.mBlocks.z = divup(params.mBlocks.y, maxBlocksY);
        params.mBlocks.y = divup(params.mBlocks.y, params.mBlocks.z);
    } else if (baseDim == 2) {
        params.mThreads = dim3(CONV2_THREADS_X, CONV2_THREADS_Y);
        params.mBlk_x   = divup(oDims[0], params.mThreads.x);
        params.mBlk_y   = divup(oDims[1], params.mThreads.y);
        params.mBlocks =
            dim3(params.mBlk_x * batchDims[2], params.mBlk_y * batchDims[3]);
        params.mBlocks.z = divup(params.mBlocks.y, maxBlocksY);
        params.mBlocks.y = divup(params.mBlocks.y, params.mBlocks.z);
    } else if (baseDim == 3) {
        params.mThreads = dim3(CONV3_CUBE_X, CONV3_CUBE_Y, CONV3_CUBE_Z);
        params.mBlk_x   = divup(oDims[0], params.mThreads.x);
        params.mBlk_y   = divup(oDims[1], params.mThreads.y);
        int blk_z       = divup(oDims[2], params.mThreads.z);
        params.mBlocks =
            dim3(params.mBlk_x * batchDims[3], params.mBlk_y, blk_z);
        params.mSharedSize = (params.mThreads.x + 2 * (fDims[0] - 1)) *
                             (params.mThreads.y + 2 * (fDims[1] - 1)) *
                             (params.mThreads.z + 2 * (fDims[2] - 1)) *
                             sizeof(T);
    }
}

template<typename T, typename aT>
void convolve_1d(conv_kparam_t& p, Param<T> out, CParam<T> sig, CParam<aT> filt,
                 const bool expand) {
    auto convolve1 = common::getKernel(
        "arrayfire::cuda::convolve1", {{convolve1_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateTypename<aT>(),
                     TemplateArg(expand)),
        {{DefineValue(MAX_CONV1_FILTER_LEN), DefineValue(CONV_THREADS)}});

    prepareKernelArgs<T>(p, out.dims, filt.dims, 1);

    size_t filterSize = filt.dims[0] * sizeof(aT);

    for (int b3 = 0; b3 < filt.dims[3]; ++b3) {
        int f3Off = b3 * filt.strides[3];

        for (int b2 = 0; b2 < filt.dims[2]; ++b2) {
            int f2Off = b2 * filt.strides[2];

            for (int b1 = 0; b1 < filt.dims[1]; ++b1) {
                int f1Off      = b1 * filt.strides[1];
                const aT* fptr = filt.ptr + (f1Off + f2Off + f3Off);

                // FIXME: case where filter array is strided
                auto constMemPtr = convolve1.getDevPtr(conv_c_name);
                convolve1.copyToReadOnly(constMemPtr,
                                         reinterpret_cast<CUdeviceptr>(fptr),
                                         filterSize);

                p.o[0] = (p.outHasNoOffset ? 0 : b1);
                p.o[1] = (p.outHasNoOffset ? 0 : b2);
                p.o[2] = (p.outHasNoOffset ? 0 : b3);
                p.s[0] = (p.inHasNoOffset ? 0 : b1);
                p.s[1] = (p.inHasNoOffset ? 0 : b2);
                p.s[2] = (p.inHasNoOffset ? 0 : b3);

                EnqueueArgs qArgs(p.mBlocks, p.mThreads, getActiveStream(),
                                  p.mSharedSize);
                convolve1(qArgs, out, sig, filt.dims[0], p.mBlk_x, p.mBlk_y,
                          p.o[0], p.o[1], p.o[2], p.s[0], p.s[1], p.s[2]);
                POST_LAUNCH_CHECK();
            }
        }
    }
}

template<typename T, typename aT>
void conv2Helper(const conv_kparam_t& p, Param<T> out, CParam<T> sig,
                 const aT* fptr, int f0, int f1, const bool expand) {
    const bool isFilterSizeLt5  = (f0 <= 5 && f1 <= 5);
    const bool isFilterGt5AndSq = (f0 == f1 && f0 > 5 && f0 < 18);

    if (!(isFilterSizeLt5 || isFilterGt5AndSq)) {
        char errMessage[256];
        snprintf(errMessage, sizeof(errMessage),
                 "\nCUDA Convolution doesn't support %dx%d kernel\n", f0, f1);
        CUDA_NOT_SUPPORTED(errMessage);
    }

    auto convolve2 = common::getKernel(
        "arrayfire::cuda::convolve2", {{convolve2_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateTypename<aT>(),
                     TemplateArg(expand), TemplateArg(f0), TemplateArg(f1)),
        {{DefineValue(MAX_CONV1_FILTER_LEN), DefineValue(CONV_THREADS),
          DefineValue(CONV2_THREADS_X), DefineValue(CONV2_THREADS_Y)}});

    // FIXME: case where filter array is strided
    auto constMemPtr = convolve2.getDevPtr(conv_c_name);
    convolve2.copyToReadOnly(constMemPtr, reinterpret_cast<CUdeviceptr>(fptr),
                             f0 * f1 * sizeof(aT));

    EnqueueArgs qArgs(p.mBlocks, p.mThreads, getActiveStream());
    convolve2(qArgs, out, sig, p.mBlk_x, p.mBlk_y, p.o[1], p.o[2], p.s[1],
              p.s[2]);
    POST_LAUNCH_CHECK();
}

template<typename T, typename aT>
void convolve_2d(conv_kparam_t& p, Param<T> out, CParam<T> sig, CParam<aT> filt,
                 const bool expand) {
    prepareKernelArgs<T>(p, out.dims, filt.dims, 2);

    for (int b3 = 0; b3 < filt.dims[3]; ++b3) {
        int f3Off = b3 * filt.strides[3];

        for (int b2 = 0; b2 < filt.dims[2]; ++b2) {
            int f2Off = b2 * filt.strides[2];

            const aT* fptr = filt.ptr + (f2Off + f3Off);

            p.o[1] = (p.outHasNoOffset ? 0 : b2);
            p.o[2] = (p.outHasNoOffset ? 0 : b3);
            p.s[1] = (p.inHasNoOffset ? 0 : b2);
            p.s[2] = (p.inHasNoOffset ? 0 : b3);

            conv2Helper<T, aT>(p, out, sig, fptr, filt.dims[0], filt.dims[1],
                               expand);
        }
    }
}

template<typename T, typename aT>
void convolve_3d(conv_kparam_t& p, Param<T> out, CParam<T> sig, CParam<aT> filt,
                 const bool expand) {
    auto convolve3 = common::getKernel(
        "arrayfire::cuda::convolve3", {{convolve3_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateTypename<aT>(),
                     TemplateArg(expand)),
        {{DefineValue(MAX_CONV1_FILTER_LEN), DefineValue(CONV_THREADS),
          DefineValue(CONV3_CUBE_X), DefineValue(CONV3_CUBE_Y),
          DefineValue(CONV3_CUBE_Z)}});

    prepareKernelArgs<T>(p, out.dims, filt.dims, 3);

    size_t filterSize = filt.dims[0] * filt.dims[1] * filt.dims[2] * sizeof(aT);

    for (int b3 = 0; b3 < filt.dims[3]; ++b3) {
        int f3Off = b3 * filt.strides[3];

        const aT* fptr = filt.ptr + f3Off;

        // FIXME: case where filter array is strided
        auto constMemPtr = convolve3.getDevPtr(conv_c_name);
        convolve3.copyToReadOnly(
            constMemPtr, reinterpret_cast<CUdeviceptr>(fptr), filterSize);

        p.o[2] = (p.outHasNoOffset ? 0 : b3);
        p.s[2] = (p.inHasNoOffset ? 0 : b3);

        EnqueueArgs qArgs(p.mBlocks, p.mThreads, getActiveStream(),
                          p.mSharedSize);
        convolve3(qArgs, out, sig, filt.dims[0], filt.dims[1], filt.dims[2],
                  p.mBlk_x, p.o[2], p.s[2]);
        POST_LAUNCH_CHECK();
    }
}

template<typename T, typename aT>
void convolve_nd(Param<T> out, CParam<T> signal, CParam<aT> filt,
                 AF_BATCH_KIND kind, int baseDim, bool expand) {
    bool callKernel = true;

    int MCFL2 = kernel::MAX_CONV2_FILTER_LEN;
    int MCFL3 = kernel::MAX_CONV3_FILTER_LEN;
    switch (baseDim) {
        case 1:
            if (filt.dims[0] > kernel::MAX_CONV1_FILTER_LEN) callKernel = false;
            break;
        case 2:
            if ((filt.dims[0] * filt.dims[1]) > (MCFL2 * MCFL2))
                callKernel = false;
            break;
        case 3:
            if ((filt.dims[0] * filt.dims[1] * filt.dims[2]) >
                (MCFL3 * MCFL3 * MCFL3))
                callKernel = false;
            break;
    }

    if (!callKernel) {
        char errMessage[256];
        snprintf(errMessage, sizeof(errMessage),
                 "\nCUDA N Dimensional Convolution doesn't support "
                 "%lldx%lldx%lld kernel\n",
                 filt.dims[0], filt.dims[1], filt.dims[2]);
        CUDA_NOT_SUPPORTED(errMessage);
    }

    conv_kparam_t param;
    for (int i = 0; i < 3; ++i) {
        param.o[i] = 0;
        param.s[i] = 0;
    }
    param.launchMoreBlocks = kind == AF_BATCH_SAME || kind == AF_BATCH_RHS;
    param.outHasNoOffset   = kind == AF_BATCH_LHS || kind == AF_BATCH_NONE;
    param.inHasNoOffset    = kind != AF_BATCH_SAME;

    switch (baseDim) {
        case 1: convolve_1d<T, aT>(param, out, signal, filt, expand); break;
        case 2: convolve_2d<T, aT>(param, out, signal, filt, expand); break;
        case 3: convolve_3d<T, aT>(param, out, signal, filt, expand); break;
    }

    POST_LAUNCH_CHECK();
}

static const int SCONV_THREADS_X = 16;
static const int SCONV_THREADS_Y = 16;

// below shared MAX_*_LEN's are calculated based on
// a maximum shared memory configuration of 48KB per block
// considering complex types as well
static const int MAX_SCONV_FILTER_LEN = 31;

template<typename T, typename aT>
void convolve2(Param<T> out, CParam<T> signal, CParam<aT> filter, int conv_dim,
               bool expand) {
    int fLen =
        filter.dims[0] * filter.dims[1] * filter.dims[2] * filter.dims[3];

    if (fLen > kernel::MAX_SCONV_FILTER_LEN) {
        // TODO call upon fft
        char errMessage[256];
        snprintf(errMessage, sizeof(errMessage),
                 "\nCUDA convolution supports max kernel size of %d\n",
                 kernel::MAX_SCONV_FILTER_LEN);
        CUDA_NOT_SUPPORTED(errMessage);
    }

    auto convolve2_separable = common::getKernel(
        "arrayfire::cuda::convolve2_separable", {{convolve_separable_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateTypename<aT>(),
                     TemplateArg(conv_dim), TemplateArg(expand),
                     TemplateArg(fLen)),
        {{DefineValue(MAX_SCONV_FILTER_LEN), DefineValue(SCONV_THREADS_X),
          DefineValue(SCONV_THREADS_Y)}});

    dim3 threads(SCONV_THREADS_X, SCONV_THREADS_Y);

    int blk_x = divup(out.dims[0], threads.x);
    int blk_y = divup(out.dims[1], threads.y);

    dim3 blocks(blk_x * signal.dims[2], blk_y * signal.dims[3]);

    // FIXME: case where filter array is strided
    auto constMemPtr = convolve2_separable.getDevPtr(sconv_c_name);
    convolve2_separable.copyToReadOnly(
        constMemPtr, reinterpret_cast<CUdeviceptr>(filter.ptr),
        fLen * sizeof(aT));

    EnqueueArgs qArgs(blocks, threads, getActiveStream());
    convolve2_separable(qArgs, out, signal, blk_x, blk_y);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
