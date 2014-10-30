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
#include <errorcodes.hpp>
#include <err_common.hpp>

#define OPENCL_NOT_SUPPORTED() do {                         \
        throw SupportError(__FILE__, __LINE__, "OPENCL");   \
    } while(0)

#define CL_TO_AF_ERROR(ERR) do {                        \
        char opencl_err_msg[1024];                      \
        snprintf(opencl_err_msg,                        \
                 sizeof(opencl_err_msg),                \
                 "OpenCL Error: %s when calling %s",    \
                 getErrorMessage(ERR.err()),            \
                 ERR.what());                           \
        AF_ERROR(opencl_err_msg,                        \
                 AF_ERR_INTERNAL);                      \
    } while(0)
