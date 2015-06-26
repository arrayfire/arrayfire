/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <stdio.h>
#include <defines.hpp>
#include <err_common.hpp>

#define CUDA_NOT_SUPPORTED() do {                       \
        throw SupportError(__FILE__, __LINE__, "CUDA"); \
    } while(0)

#define CUDA_CHECK(fn) do {                     \
        cudaError_t _cuda_error = fn;           \
        if (_cuda_error != cudaSuccess) {       \
            char cuda_err_msg[1024];            \
            snprintf(cuda_err_msg,                \
                     sizeof(cuda_err_msg),      \
                     "CUDA Error (%d): %s\n",   \
                     (int)(_cuda_error),        \
                     cudaGetErrorString(        \
                         _cuda_error));         \
                                                \
            AF_ERROR(cuda_err_msg,              \
                     AF_ERR_INTERNAL);          \
        }                                       \
    } while(0)

