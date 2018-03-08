/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <chrono>

#define AF_BENCH_TIMER_START()  \
    auto start = std::chrono::high_resolution_clock::now();

#define AF_BENCH_TIMER_STOP()   \
    af::sync();                                                         \
    auto end = std::chrono::high_resolution_clock::now();               \
    auto elapsed_seconds =                                              \
            std::chrono::duration_cast<std::chrono::duration<double>>(  \
                    end - start);                                       \
    state.SetIterationTime(elapsed_seconds.count());

