/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <platform.hpp>
#include <queue.hpp>
#include <err_cpu.hpp>

#ifndef NDEBUG

#define POST_LAUNCH_CHECK() do {                        \
        getQueue().sync();                              \
    } while(0)                                          \

#else

#define POST_LAUNCH_CHECK() //no-op

#endif

#define ENQUEUE(...)                        \
    do {                                    \
        getQueue().enqueue(__VA_ARGS__);    \
        POST_LAUNCH_CHECK();                \
    } while(0)
