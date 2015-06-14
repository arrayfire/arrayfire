/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined(WITH_CUDA_LINEAR_ALGEBRA)
#pragma once

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdio.h>
#include <defines.hpp>
#include <err_common.hpp>

namespace cusolver
{

    const char * errorString(cusolverStatus_t err);
    cusolverDnHandle_t getDnHandle();
}

#define CUSOLVER_CHECK(fn) do {                     \
        cusolverStatus_t _error = fn;               \
        if (_error != CUSOLVER_STATUS_SUCCESS) {    \
            char _err_msg[1024];                    \
            snprintf(_err_msg,                      \
                     sizeof(_err_msg),              \
                     "CUBLAS Error (%d): %s\n",     \
                     (int)(_error),                 \
                     cusolver::errorString(         \
                         _error));                  \
                                                    \
            AF_ERROR(_err_msg,                      \
                     AF_ERR_INTERNAL);              \
        }                                           \
    } while(0)

#endif
