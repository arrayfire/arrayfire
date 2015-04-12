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

// FIXME: Add a special flag for debug
#ifndef NDEBUG

#define POST_LAUNCH_CHECK() do {                \
        CUDA_CHECK(cudaDeviceSynchronize());    \
    } while(0)                                  \

#else

#define POST_LAUNCH_CHECK() do {                \
        CUDA_CHECK(cudaPeekAtLastError());      \
    } while(0)                                  \

#endif
