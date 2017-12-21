/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <err_common.hpp>

#define CPU_NOT_SUPPORTED(message) do {                 \
        throw SupportError(__PRETTY_FUNCTION__,         \
                __AF_FILENAME__, __LINE__, message);    \
    } while(0)
