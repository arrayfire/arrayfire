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
#include <cusparse_v2.h>

namespace cusparse {

    const char * errorString(cusparseStatus_t err);
    cusparseHandle_t getHandle();
}


#define CUSPARSE_CHECK(fn) do {                         \
        cusparseStatus_t _error = fn;                   \
        if (_error != CUSPARSE_STATUS_SUCCESS) {        \
            char _err_msg[1024];                        \
            snprintf(_err_msg, sizeof(_err_msg),        \
                     "CUSPARSE Error (%d): %s\n",       \
                     (int)(_error),                     \
                     cusparse::errorString( _error));   \
                                                        \
            AF_ERROR(_err_msg, AF_ERR_INTERNAL);        \
        }                                               \
    } while(0)
