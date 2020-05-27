/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/err_common.hpp>

#include <cuda.h>

#include <cstdio>

#define CU_CHECK(fn)                                                      \
    do {                                                                  \
        CUresult res = fn;                                                \
        if (res == CUDA_SUCCESS) break;                                   \
        char cu_err_msg[1024];                                            \
        const char* cu_err_name;                                          \
        const char* cu_err_string;                                        \
        cuGetErrorName(res, &cu_err_name);                                \
        cuGetErrorString(res, &cu_err_string);                            \
        snprintf(cu_err_msg, sizeof(cu_err_msg), "CU Error %s(%d): %s\n", \
                 cu_err_name, (int)(res), cu_err_string);                 \
        AF_ERROR(cu_err_msg, AF_ERR_INTERNAL);                            \
    } while (0)
