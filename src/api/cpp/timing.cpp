/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <math.h>
#include <af/device.h>
#include <af/timing.h>
#include <algorithm>
#include <vector>

using namespace af;

// get current time
static inline timer time_now(void) {
#if defined(OS_WIN)
    timer time;
    QueryPerformanceCounter(&time.val);
#elif defined(OS_MAC)
    timer time = {mach_absolute_time()};
#elif defined(OS_LNX)
    timer time;
    gettimeofday(&time.val, NULL);
#endif
    return time;
}

// absolute difference between two times (in seconds)
static inline double time_seconds(timer start, timer end) {
#if defined(OS_WIN)
    if (start.val.QuadPart > end.val.QuadPart) {
        timer temp = end;
        end        = start;
        start      = temp;
    }
    timer system_freq;
    QueryPerformanceFrequency(&system_freq.val);
    return (double)(end.val.QuadPart - start.val.QuadPart) /
           system_freq.val.QuadPart;
#elif defined(OS_MAC)
    if (start.val > end.val) {
        timer temp = start;
        start      = end;
        end        = temp;
    }
    // calculate platform timing epoch
    thread_local mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    double nano = (double)info.numer / (double)info.denom;
    return (end.val - start.val) * nano * 1e-9;
#elif defined(OS_LNX)
    struct timeval elapsed;
    timersub(&start.val, &end.val, &elapsed);
    long sec  = elapsed.tv_sec;
    long usec = elapsed.tv_usec;
    double t  = sec + usec * 1e-6;
    return t >= 0 ? t : -t;
#endif
}

namespace af {

thread_local timer _timer_;

timer timer::start() { return _timer_ = time_now(); }
double timer::stop(timer start) { return time_seconds(start, time_now()); }
double timer::stop() { return time_seconds(_timer_, time_now()); }

double timeit(void (*fn)()) {
    // parameters
    static const int trials      = 10;  // trial runs
    static const int s_trials    = 5;   // trial runs
    static const double min_time = 1;   // seconds

    std::vector<double> sample_times(s_trials);

    // estimate time for a few samples
    for (int i = 0; i < s_trials; ++i) {
        sync();
        timer start = timer::start();
        fn();
        sync();
        sample_times[i] = timer::stop(start);
    }

    // Sort sample times and select the median time
    std::sort(sample_times.begin(), sample_times.end());

    double median_time = sample_times[s_trials / 2];

    // Run a bunch of batches of fn
    // Each batch runs trial runs before sync
    // If trials * median_time < min time,
    //   then run (min time / (trials * median_time)) batches
    // else
    //   run 1 batch
    int batches     = (int)ceilf(min_time / (trials * median_time));
    double run_time = 0;

    for (int b = 0; b < batches; b++) {
        timer start = timer::start();
        for (int i = 0; i < trials; ++i) fn();
        sync();
        run_time += timer::stop(start) / trials;
    }
    return run_time / batches;
}

}  // namespace af
