/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#ifndef NDEBUG

#define CL_DEBUG_FINISH(Q) Q.finish()

#else

#include <platform.hpp>

#define CL_DEBUG_FINISH(Q)                       \
    do {                                         \
        if (synchronize_calls()) { Q.finish(); } \
    } while (false);

#endif
