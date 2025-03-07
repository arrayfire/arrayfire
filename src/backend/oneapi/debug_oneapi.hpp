/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <platform.hpp>

#ifndef NDEBUG

#define ONEAPI_DEBUG_FINISH(Q) Q.wait_and_throw()

#else

#define ONEAPI_DEBUG_FINISH(Q)                                   \
    do {                                                         \
        if (oneapi::synchronize_calls()) { Q.wait_and_throw(); } \
    } while (false);

#endif
