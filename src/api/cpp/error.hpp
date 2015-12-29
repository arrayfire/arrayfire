/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/exception.h>
#include <defines.hpp>

#define AF_THROW(fn) do {                               \
        af_err __err = fn;                              \
        if (__err == AF_SUCCESS) break;                 \
        throw af::exception(__AF_FILENAME__, __LINE__, __err); \
    } while(0)

#define AF_THROW_ERR(__msg, __err) do {                         \
        throw af::exception(__msg, __AF_FILENAME__, __LINE__, __err);  \
    } while(0)
