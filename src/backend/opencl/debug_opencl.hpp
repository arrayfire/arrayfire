/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

// - Include platform.hpp where this macro is used
//   for functions synchronize_calls()
// - Include cl2hpp.hpp for cl::CommandQueue::finish() method

#ifndef NDEBUG
#define CL_DEBUG_FINISH(Q) Q.finish()
#else
#define CL_DEBUG_FINISH(Q)                       \
    do {                                         \
        if (synchronize_calls()) { Q.finish(); } \
    } while (false);
#endif
