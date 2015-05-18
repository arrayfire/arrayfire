/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "types.h"

#define MATH_BASIC(fn, T)                       \
    __device__ T ___##fn(T a)                   \
    {                                           \
        return fn##f((float)a);                 \
    }                                           \


#define MATH(fn)                                \
    MATH_BASIC(fn, float)                       \
    MATH_BASIC(fn, int)                         \
    MATH_BASIC(fn, uint)                        \
    MATH_BASIC(fn, char)                        \
    MATH_BASIC(fn, uchar)                       \
    __device__ double ___##fn(double a)         \
    {                                           \
        return fn(a);                           \
    }                                           \


MATH(sin)
MATH(cos)
MATH(tan)

MATH(asin)
MATH(acos)
MATH(atan)

#define ATAN2(T)                                \
    __device__ T ___atan2(T x, T y)             \
    {                                           \
        return atan2((float)x, (float)y);       \
    }                                           \

ATAN2(float)
ATAN2(int)
ATAN2(uint)
ATAN2(char)
ATAN2(uchar)

__device__ double ___atan2(double x, double y)
{
    return atan2(x, y);
}
