/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#ifndef __PROGRESS_H
#define __PROGRESS_H

#include <cmath>
#include <algorithm>

static bool progress(unsigned iter_curr, af::timer t, double time_total)
{
    static unsigned iter_prev = 0;
    static double time_prev = 0;
    static double max_rate = 0;

    af::sync();
    double time_curr = af::timer::stop(t);

    if ((time_curr - time_prev) < 1) return true;

    double rate = (iter_curr - iter_prev) / (time_curr - time_prev);
    printf("  iterations per second: %.0f   (progress %.0f%%)\n",
            rate, 100.0f * time_curr / time_total);

    max_rate = std::max(max_rate, rate);

    iter_prev = iter_curr;
    time_prev = time_curr;


    if (time_curr < time_total) return true;

    printf(" ### vortex %f iterations per second (max)\n", max_rate);
    return false;
}

#endif
