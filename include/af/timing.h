/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>

#ifdef __cplusplus

#if defined(_WIN32) || defined(_MSC_VER)
  #include <windows.h>
#elif defined(__APPLE__) && defined(__MACH__)
  // http://developer.apple.com/qa/qa2004/qa1398.html
  #include <mach/mach_time.h>
#else // Linux
  #ifndef AF_DOC
    #include <sys/time.h>
  #endif
#endif

namespace af {

/// Internal timer object
    typedef struct timer {
    #if defined(_WIN32) || defined(_MSC_VER)
      LARGE_INTEGER val;
    #elif defined(__APPLE__) && defined(__MACH__)
      uint64_t val;
    #else // Linux
      struct timeval val;
    #endif

    AFAPI static timer start();

    AFAPI static double stop();
    AFAPI static double stop(timer start);

} timer;

AFAPI double timeit(void(*fn)());
}

#endif
