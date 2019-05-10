/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/HandleBase.hpp>
#include <common/defines.hpp>
#include <common/err_common.hpp>
#include <cuda_runtime.h>
#include <cusolverDn.h>

namespace cuda {

using SolveHandle = cusolverDnHandle_t;

const char* errorString(cusolverStatus_t err);

#define CUSOLVER_CHECK(fn)                                                  \
    do {                                                                    \
        cusolverStatus_t _error = fn;                                       \
        if (_error != CUSOLVER_STATUS_SUCCESS) {                            \
            char _err_msg[1024];                                            \
            snprintf(_err_msg, sizeof(_err_msg), "CUBLAS Error (%d): %s\n", \
                     (int)(_error), errorString(_error));                   \
                                                                            \
            AF_ERROR(_err_msg, AF_ERR_INTERNAL);                            \
        }                                                                   \
    } while (0)

CREATE_HANDLE(cusolverDnHandle, cusolverDnHandle_t, cusolverDnCreate,
              cusolverDnDestroy, CUSOLVER_CHECK);

}  // namespace cuda
