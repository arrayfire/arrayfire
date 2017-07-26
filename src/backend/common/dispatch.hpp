/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <cmath>

#define divup(a, b) (((a)+(b)-1)/(b))

unsigned nextpow2(unsigned x);

// isPrime & greatestPrimeFactor are tailored after
// itk::Math::{IsPrimt, GreatestPrimeFactor}
template <typename T>
inline bool isPrime(T n)
{
    if( n <= 1 )
        return false;

    const T last = (T)std::sqrt( (double)n );
    for (T x=2; x<=last; ++x)
    {
        if (n%x == 0)
            return false;
    }

    return true;
}

template <typename T>
inline T greatestPrimeFactor(T n)
{
    T v = 2;

    while (v <= n)
    {
        if (n % v == 0 && isPrime(v))
            n /= v;
        else
            v += 1;
    }

    return v;
}
