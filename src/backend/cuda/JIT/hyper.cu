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
    MATH_BASIC(fn, uintl)                       \
    MATH_BASIC(fn, intl)                        \
    __device__ double ___##fn(double a)         \
    {                                           \
        return fn(a);                           \
    }                                           \


MATH(sinh)
MATH(cosh)
MATH(tanh)

MATH(asinh)
MATH(acosh)
MATH(atanh)
