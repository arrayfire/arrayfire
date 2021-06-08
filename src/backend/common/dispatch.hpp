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
#include <algorithm>
#include <cmath>

#define divup(a, b) (((a) + (b)-1) / (b))

unsigned nextpow2(unsigned x);

// isPrime & greatestPrimeFactor are tailored after
// itk::Math::{IsPrimt, GreatestPrimeFactor}
template<typename T>
inline bool isPrime(T n) {
    if (n <= 1) return false;

    const T last = (T)std::sqrt((double)n);
    for (T x = 2; x <= last; ++x) {
        if (n % x == 0) return false;
    }

    return true;
}

template<typename T>
inline T greatestPrimeFactor(T n) {
    T v = 2;

    while (v <= n) {
        if (n % v == 0 && isPrime(v))
            n /= v;
        else
            v += 1;
    }

    return v;
}

// For OPENCL, the dimensions of local are returned
// usage: cl::NDRange local = bestBlockSize<cl::NDRange>(dims, WG)
// For CUDA, the dimensions of 1 block are returned
// usage: dim3 block = bestBlockSize<dim3>(dims, 32);
// The parameter dims can have any type as long as it is convertable to unsigned

// Remark: The bestBlockSize is only best for independent element operations, as
// are: copying, scaling, math on independent elements, ...
// Since vector dimensions can be returned, it is NOT USABLE FOR BLOCK
// OPERATIONS, as are: matmul, etc.
template<typename Tout, typename Tin>
Tout bestBlockSize(const Tin dims[4], unsigned warp) {
    const unsigned d0         = static_cast<unsigned>(dims[0]);
    const unsigned d1         = static_cast<unsigned>(dims[1]);
    const unsigned d2         = static_cast<unsigned>(dims[2]);
    const unsigned OCC        = 3;
    const unsigned elements   = d0 * d1;
    const unsigned minThreads = warp / 4;  // quarter wave
    const unsigned maxThreads =
        std::min(warp * 4, divup(elements * warp, 16384U) * minThreads);

    const unsigned threads0 =
#ifdef AF_OPENCL
        (d0 < warp) ? d0 :
#endif
        (d1 == 1) ? warp * 4
        : (maxThreads >= 128) && (!(d0 & (128 - 1)) || (d0 > OCC * (128 - 1)))
            ? 128
        : (maxThreads >= 64) && (!(d0 & (64 - 1)) || (d0 > OCC * (64 - 1)))
            ? 64
            : warp;

    const unsigned threads1 =
        (threads0 <= maxThreads / 128) &&
                (!(d1 & (128 - 1)) || (d1 > OCC * (128 - 1)))
            ? 128
        : (threads0 <= maxThreads / 64) &&
                (!(d1 & (64 - 1)) || (d1 > OCC * (64 - 1)))
            ? 64
        : (threads0 <= maxThreads / 32) &&
                (!(d1 & (32 - 1)) || (d1 > OCC * (32 - 1)))
            ? 32
        : (threads0 <= maxThreads / 16) &&
                (!(d1 & (16 - 1)) || (d1 > OCC * (16 - 1)))
            ? 16
        : (threads0 <= maxThreads / 8) &&
                (!(d1 & (8 - 1)) || (d1 > OCC * (8 - 1)))
            ? 8
        : (threads0 <= maxThreads / 4) &&
                (!(d1 & (4 - 1)) || (d1 > OCC * (4 - 1)))
            ? 4
        : (threads0 <= maxThreads / 2) &&
                (!(d1 & (2 - 1)) || (d1 > OCC * (2 - 1)))
            ? 2
            : 1;

    const unsigned threads01 = threads0 * threads1;
    if (d2 == 1 || threads01 * 2 > maxThreads) return Tout(threads0, threads1);

    const unsigned threads2 =
        (threads01 <= maxThreads / 64) && !(d2 & (64 - 1))   ? 64
        : (threads01 <= maxThreads / 32) && !(d2 & (32 - 1)) ? 32
        : (threads01 <= maxThreads / 16) && !(d2 & (16 - 1)) ? 16
        : (threads01 <= maxThreads / 8) && !(d2 & (8 - 1))   ? 8
        : (threads01 <= maxThreads / 4) && !(d2 & (4 - 1))   ? 4
        : (threads01 <= maxThreads / 2) && !(d2 & (2 - 1))   ? 2
                                                             : 1;
    return Tout(threads0, threads1, threads2);
}