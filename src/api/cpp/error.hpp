/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/exception.h>

#define AF_THROW(fn) do {                               \
        af_err __err = fn;                              \
        if (__err == AF_SUCCESS) break;                 \
        throw af::exception(__FILE__, __LINE__, __err); \
    } while(0)

#define AF_THROW_MSG(__msg, __err) do {                         \
        if (__err == AF_SUCCESS) break;                         \
        throw af::exception(__msg, __FILE__, __LINE__, __err);  \
    } while(0);

#define THROW(__err) throw af::exception(__FILE__, __LINE__, __err)
