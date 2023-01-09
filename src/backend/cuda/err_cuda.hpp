/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <common/defines.hpp>
#include <common/err_common.hpp>
#include <stdio.h>

#define CUDA_NOT_SUPPORTED(message)                                         \
    do {                                                                    \
        throw SupportError(__AF_FUNC__, __AF_FILENAME__, __LINE__, message, \
                           boost::stacktrace::stacktrace());                \
    } while (0)

#define CU_CHECK(fn)                                                          \
    do {                                                                      \
        CUresult res = fn;                                                    \
        if (res == CUDA_SUCCESS) break;                                       \
        char cu_err_msg[1024];                                                \
        const char* cu_err_name;                                              \
        const char* cu_err_string;                                            \
        CUresult nameErr, strErr;                                             \
        nameErr = cuGetErrorName(res, &cu_err_name);                          \
        strErr  = cuGetErrorString(res, &cu_err_string);                      \
        if (nameErr == CUDA_SUCCESS && strErr == CUDA_SUCCESS) {              \
            snprintf(cu_err_msg, sizeof(cu_err_msg), "CU Error %s(%d): %s\n", \
                     cu_err_name, (int)(res), cu_err_string);                 \
            AF_ERROR(cu_err_msg, AF_ERR_INTERNAL);                            \
        } else {                                                              \
            AF_ERROR("CU Unknown error.\n", AF_ERR_INTERNAL);                 \
        }                                                                     \
    } while (0)

#define CUDA_CHECK(fn)                                               \
    do {                                                             \
        cudaError_t _cuda_error = fn;                                \
        if (_cuda_error != cudaSuccess) {                            \
            char cuda_err_msg[1024];                                 \
            snprintf(cuda_err_msg, sizeof(cuda_err_msg),             \
                     "CUDA Error (%d): %s\n", (int)(_cuda_error),    \
                     cudaGetErrorString(cudaGetLastError()));        \
                                                                     \
            if (_cuda_error == cudaErrorMemoryAllocation) {          \
                AF_ERROR(cuda_err_msg, AF_ERR_NO_MEM);               \
            } else if (_cuda_error == cudaErrorDevicesUnavailable) { \
                AF_ERROR(cuda_err_msg, AF_ERR_DRIVER);               \
            } else {                                                 \
                AF_ERROR(cuda_err_msg, AF_ERR_INTERNAL);             \
            }                                                        \
        }                                                            \
    } while (0)
