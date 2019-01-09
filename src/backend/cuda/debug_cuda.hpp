/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <err_cuda.hpp>
#include <platform.hpp>
#include <thrust/system/cuda/detail/par.h>
#include <thrust/version.h>
#include <ThrustAllocator.cuh>

namespace cuda {
template <typename T>
using ThrustVector = thrust::device_vector<T, cuda::ThrustAllocator<T>>;
}

#define THRUST_STREAM thrust::cuda::par.on(cuda::getActiveStream())

#if THRUST_MAJOR_VERSION >= 1 && THRUST_MINOR_VERSION >= 8

#define THRUST_SELECT(fn, ...) fn(THRUST_STREAM, __VA_ARGS__)
#define THRUST_SELECT_OUT(res, fn, ...) res = fn(THRUST_STREAM, __VA_ARGS__)

#else

#define THRUST_SELECT(fn, ...)                                      \
    do {                                                            \
        CUDA_CHECK(cudaStreamSynchronize(cuda::getActiveStream())); \
        fn(__VA_ARGS__);                                            \
    } while (0)

#define THRUST_SELECT_OUT(res, fn, ...)                             \
    do {                                                            \
        CUDA_CHECK(cudaStreamSynchronize(cuda::getActiveStream())); \
        res = fn(__VA_ARGS__);                                      \
    } while (0)

#endif

#define CUDA_LAUNCH_SMEM(fn, blks, thrds, smem_size, ...) \
    fn<<<blks, thrds, smem_size, cuda::getActiveStream()>>>(__VA_ARGS__)

#define CUDA_LAUNCH(fn, blks, thrds, ...) \
    CUDA_LAUNCH_SMEM(fn, blks, thrds, 0, __VA_ARGS__)

// FIXME: Add a special flag for debug
#ifndef NDEBUG

#define POST_LAUNCH_CHECK() \
    do { CUDA_CHECK(cudaStreamSynchronize(cuda::getActiveStream())); } while (0)

#else

#define POST_LAUNCH_CHECK()                                             \
    do {                                                                \
        if (cuda::synchronize_calls()) {                                \
            CUDA_CHECK(cudaStreamSynchronize(cuda::getActiveStream())); \
        } else {                                                        \
            CUDA_CHECK(cudaPeekAtLastError());                          \
        }                                                               \
    } while (0)

#endif
