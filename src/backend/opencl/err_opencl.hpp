/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/err_common.hpp>

#define OPENCL_NOT_SUPPORTED(message)                                       \
    do {                                                                    \
        throw SupportError(__AF_FUNC__, __AF_FILENAME__, __LINE__, message, \
                           boost::stacktrace::stacktrace());                \
    } while (0)
