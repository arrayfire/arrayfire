/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/timing.h>
#include <af/device.h>
#include <math.h>
#include <algorithm>

using namespace af;

// get current time
static inline timer time_now(void)
{
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
static inline double time_seconds(timer start, timer end)
{
#if defined(OS_WIN)
    if (start.val.QuadPart > end.val.QuadPart) {
        timer temp = end;
        end = start;
        start = temp;
    }
    timer system_freq;
    QueryPerformanceFrequency(&system_freq.val);
    return (double)(end.val.QuadPart - start.val.QuadPart) / system_freq.val.QuadPart;
#elif defined(OS_MAC)
    if (start.val > end.val) {
        timer temp = start;
        start = end;
        end   = temp;
    }
    // calculate platform timing epoch
    static mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    double nano = (double)info.numer / (double)info.denom;
    return (end.val - start.val) * nano * 1e-9;
#elif defined(OS_LNX)
    struct timeval elapsed;
    timersub(&start.val, &end.val, &elapsed);
    long sec = elapsed.tv_sec;
    long usec = elapsed.tv_usec;
    double t = sec + usec * 1e-6;
    return t >= 0 ? t : -t;
#endif
}


namespace af {

static timer _timer_;

AFAPI timer timer::start()
{
    return _timer_ = time_now();
}
AFAPI double timer::stop(timer start)
{
    return time_seconds(start, time_now());
}
AFAPI double timer::stop()
{
    return time_seconds(_timer_, time_now());
}

AFAPI double timeit(void(*fn)())
{
    // parameters
    int sample_trials = 3;
    double min_time = 1;

    // estimate time for a few samples
    double sample_time = 1e99; // INF
    for (int i = 0; i < sample_trials; ++i) {
        sync();
        timer start = timer::start();
        fn();
        sync();
        sample_time = std::min(sample_time, timer::stop(start));
    }

    double seconds = std::max(sample_time, min_time); // at least minimum time
    double elapsed = 0;
    while (elapsed + sample_time < seconds) {
        int r = ceilf((seconds - elapsed) / sample_time);
        timer start = timer::start();
        for (int i = 0; i < r; ++i)
            fn();
        sync();
        double t = timer::stop(start);
        elapsed += t;
        sample_time = std::min(sample_time, t / r);
    }
    return sample_time;
}

} // namespace af
