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
#include <cublas_v2.h>

namespace cuda {

using BlasHandle = cublasHandle_t;

const char* errorString(cublasStatus_t err);

#define CUBLAS_CHECK(fn)                                                    \
    do {                                                                    \
        cublasStatus_t _error = fn;                                         \
        if (_error != CUBLAS_STATUS_SUCCESS) {                              \
            char _err_msg[1024];                                            \
            snprintf(_err_msg, sizeof(_err_msg), "CUBLAS Error (%d): %s\n", \
                     (int)(_error), errorString(_error));                   \
                                                                            \
            AF_ERROR(_err_msg, AF_ERR_INTERNAL);                            \
        }                                                                   \
    } while (0)

CREATE_HANDLE(cublasHandle, cublasHandle_t, cublasCreate, cublasDestroy,
              CUBLAS_CHECK);

}  // namespace cuda
