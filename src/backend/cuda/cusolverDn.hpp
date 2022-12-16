/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/unique_handle.hpp>
#include <cusolverDn.h>

DEFINE_HANDLER(cusolverDnHandle_t, cusolverDnCreate, cusolverDnDestroy);

namespace arrayfire {
namespace cuda {

const char* errorString(cusolverStatus_t err);

#define CUSOLVER_CHECK(fn)                                                    \
    do {                                                                      \
        cusolverStatus_t _error = fn;                                         \
        if (_error != CUSOLVER_STATUS_SUCCESS) {                              \
            char _err_msg[1024];                                              \
            snprintf(_err_msg, sizeof(_err_msg), "CUSOLVER Error (%d): %s\n", \
                     (int)(_error), arrayfire::cuda::errorString(_error));    \
                                                                              \
            AF_ERROR(_err_msg, AF_ERR_INTERNAL);                              \
        }                                                                     \
    } while (0)

}  // namespace cuda
}  // namespace arrayfire
