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
