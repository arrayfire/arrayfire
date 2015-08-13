/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <platform.hpp>
#include <err_cuda.hpp>
#include <thrust/version.h>
#include <thrust/system/cuda/detail/par.h>

#define THRUST_STREAM thrust::cuda::par.on(cuda::getStream(cuda::getActiveDeviceId()))

#if THRUST_MAJOR_VERSION>=1 && THRUST_MINOR_VERSION>=8

#define THRUST_SELECT(fn, ...) fn(THRUST_STREAM, __VA_ARGS__)
#define THRUST_SELECT_OUT(res, fn, ...) res = fn(THRUST_STREAM, __VA_ARGS__)

#else

#define THRUST_SELECT(fn, ...) \
    do {                          \
        CUDA_CHECK(cudaStreamSynchronize(cuda::getStream(cuda::getActiveDeviceId()))); \
        fn(__VA_ARGS__);       \
    } while(0)

#define THRUST_SELECT_OUT(res, fn, ...) \
    do {                          \
        CUDA_CHECK(cudaStreamSynchronize(cuda::getStream(cuda::getActiveDeviceId()))); \
        res = fn(__VA_ARGS__);       \
    } while(0)

#endif

#define CUDA_LAUNCH_SMEM(fn, blks, thrds, smem_size, ...) \
	fn<<<blks, thrds, smem_size, cuda::getStream(cuda::getActiveDeviceId())>>>(__VA_ARGS__)

#define CUDA_LAUNCH(fn, blks, thrds, ...) \
	CUDA_LAUNCH_SMEM(fn, blks, thrds, 0, __VA_ARGS__)

// FIXME: Add a special flag for debug
#ifndef NDEBUG

#define POST_LAUNCH_CHECK() do {                        \
        CUDA_CHECK(cudaStreamSynchronize(cuda::getStream(cuda::getActiveDeviceId()))); \
    } while(0)                                          \

#else

#define POST_LAUNCH_CHECK() do {                \
        CUDA_CHECK(cudaPeekAtLastError());      \
    } while(0)                                  \

#endif
