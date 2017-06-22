/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <cstdio>
#include <errorcodes.hpp>
#include <common/err_common.hpp>
#include <platform.hpp>
#include <types.hpp>

#define OPENCL_NOT_SUPPORTED() do {                     \
        throw SupportError(__PRETTY_FUNCTION__,         \
                __AF_FILENAME__, __LINE__, "OpenCL");   \
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
