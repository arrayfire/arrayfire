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
#include <platform.hpp>
#include <types.hpp>

#define OPENCL_NOT_SUPPORTED() do {                         \
        throw SupportError(__FILE__, __LINE__, "OPENCL");   \
    } while(0)

#define CL_TO_AF_ERROR(ERR) do {                                \
        char opencl_err_msg[1024];                              \
        snprintf(opencl_err_msg,                                \
                 sizeof(opencl_err_msg),                        \
                 "OpenCL Error: %s when calling %s",            \
                 getErrorMessage(ERR.err()).c_str(),            \
                 ERR.what());                                   \
        if (ERR.err() == CL_MEM_OBJECT_ALLOCATION_FAILURE) {    \
            AF_ERROR(opencl_err_msg, AF_ERR_NO_MEM);            \
        } else {                                                \
            AF_ERROR(opencl_err_msg,                            \
                     AF_ERR_INTERNAL);                          \
        }                                                       \
    } while(0)

namespace opencl
{
    template <typename T>
    void verifyDoubleSupport()
    {
        if ((std::is_same<T, double>::value ||
             std::is_same<T, cdouble>::value) &&
            !isDoubleSupported(getActiveDeviceId())) {
            AF_ERROR("Double precision not supported", AF_ERR_NO_DBL);
        }
    }
}
