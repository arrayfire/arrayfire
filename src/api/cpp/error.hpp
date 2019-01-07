/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/defines.hpp>
#include <af/device.h>
#include <af/exception.h>

#define AF_THROW(fn)                                                          \
    do {                                                                      \
        af_err __err = fn;                                                    \
        if (__err == AF_SUCCESS) break;                                       \
        char *msg = NULL;                                                     \
        af_get_last_error(&msg, NULL);                                        \
        af::exception ex(msg, __PRETTY_FUNCTION__, __AF_FILENAME__, __LINE__, \
                         __err);                                              \
        af_free_host(msg);                                                    \
        throw ex; /* NOLINT(misc-throw-by-value-catch-by-reference)*/         \
    } while (0)

#define AF_THROW_ERR(__msg, __err)                                       \
    do {                                                                 \
        throw af::exception(__msg, __PRETTY_FUNCTION__, __AF_FILENAME__, \
                            __LINE__, __err);                            \
    } while (0)
