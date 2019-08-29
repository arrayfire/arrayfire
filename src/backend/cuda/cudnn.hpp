/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/defines.hpp>
#include <cudnn.h>

namespace cuda {

const char *errorString(cudnnStatus_t err);

#define CUDNN_CHECK(fn)                                                    \
    do {                                                                   \
        cudnnStatus_t _error = (fn);                                       \
        if (_error != CUDNN_STATUS_SUCCESS) {                              \
            char _err_msg[1024];                                           \
            snprintf(_err_msg, sizeof(_err_msg), "CUDNN Error (%d): %s\n", \
                     (int)(_error), errorString(_error));                  \
                                                                           \
            AF_ERROR(_err_msg, AF_ERR_INTERNAL);                           \
        }                                                                  \
    } while (0)

}
