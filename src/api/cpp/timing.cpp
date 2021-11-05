/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/device.h>
#include <af/timing.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

using namespace af;

// get current time
static inline timer time_now() {
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
    struct timeval elapsed {};
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
    // Minimum target duration to limit impact of clock precision
    constexpr double targetDurationPerTest = 0.050;
    // samples during which the nr of cycles are determined to obtain target
    // duration
    constexpr int testSamples = 2;
    // cycles needed to include CPU-GPU overlapping (if present)
    constexpr int minCycles = 3;
    // initial cycles used for the test samples
    int cycles = minCycles;
    // total number of real samples taken, of which the median is returned
    constexpr int nrSamples = 10;

    std::array<double, nrSamples> X;
    for (int s = -testSamples; s < nrSamples; ++s) {
        af::sync();
        af::timer start = af::timer::start();
        for (int i = cycles; i > 0; --i) { fn(); }
        af::sync();
        const double time = af::timer::stop(start);
        if (s >= 0) {
            // real sample, so store it for later processing
            X[s] = time;
        } else {
            // test sample, so improve nr cycles
            cycles = std::max(
                minCycles,
                static_cast<int>(trunc(targetDurationPerTest / time * cycles)));
        };
    }
    std::sort(X.begin(), X.end());
    // returns the median (iso of mean), to limit impact of outliers
    return X[nrSamples / 2] / cycles;
}

}  // namespace af
