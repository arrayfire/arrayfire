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
#include <err_common.hpp>
#include <defines.hpp>
#include <cublas_v2.h>


namespace cublas {

    const char * errorString(cublasStatus_t err);
    cublasHandle_t getHandle();
}


#define CUBLAS_CHECK(fn) do {                   \
        cublasStatus_t _error = fn;             \
        if (_error != CUBLAS_STATUS_SUCCESS) {  \
            char _err_msg[1024];                \
            snprintf(_err_msg,                  \
                     sizeof(_err_msg),          \
                     "CUBLAS Error (%d): %s\n", \
                     (int)(_error),             \
                     cublas::errorString(       \
                         _error));              \
                                                \
            AF_ERROR(_err_msg,                  \
                     AF_ERR_INTERNAL);          \
        }                                       \
    } while(0)
