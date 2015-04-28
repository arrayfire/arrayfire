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
#include <cuda.h> // Need this for CUDA_VERSION
#include <cufft.h>

static const char * _cufftGetResultString(cufftResult res)
{
    switch (res)
    {
        case CUFFT_SUCCESS:
            return "cuFFT: success";

        case CUFFT_INVALID_PLAN:
            return "cuFFT: invalid plan handle passed";

        case CUFFT_ALLOC_FAILED:
            return "cuFFT: resources allocation failed";

        case CUFFT_INVALID_TYPE:
            return "cuFFT: invalid type (deprecated)";

        case CUFFT_INVALID_VALUE:
            return "cuFFT: invalid parameters passed to cuFFT API";

        case CUFFT_INTERNAL_ERROR:
            return "cuFFT: internal error detected using cuFFT";

        case CUFFT_EXEC_FAILED:
            return "cuFFT: FFT execution failed";

        case CUFFT_SETUP_FAILED:
            return "cuFFT: library initialization failed";

        case CUFFT_INVALID_SIZE:
            return "cuFFT: invalid size parameters passed";

        case CUFFT_UNALIGNED_DATA:
            return "cuFFT: unaligned data (deprecated)";

        case CUFFT_INCOMPLETE_PARAMETER_LIST:
            return "cuFFT: call is missing parameters";

        case CUFFT_INVALID_DEVICE:
            return "cuFFT: plan execution different than plan creation";

        case CUFFT_PARSE_ERROR:
            return "cuFFT: plan parse error";

        case CUFFT_NO_WORKSPACE:
            return "cuFFT: no workspace provided";

#if CUDA_VERSION >= 6050
        case CUFFT_NOT_IMPLEMENTED:
            return "cuFFT: not implemented";

        case CUFFT_LICENSE_ERROR:
            return "cuFFT: license error";
#endif
    }

    return "cuFFT: unknown error";
}

#define CUFFT_CHECK(fn) do {                    \
        cufftResult _cufft_res = fn;            \
        if (_cufft_res != CUFFT_SUCCESS) {      \
            char cufft_res_msg[1024];           \
            snprintf(cufft_res_msg,             \
                     sizeof(cufft_res_msg),     \
                     "cuFFT Error (%d): %s\n",  \
                     (int)(_cufft_res),         \
                     _cufftGetResultString(     \
                         _cufft_res));          \
                                                \
            AF_ERROR(cufft_res_msg,             \
                     AF_ERR_INTERNAL);          \
        }                                       \
    } while(0)
