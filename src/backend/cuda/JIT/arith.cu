/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "types.h"

#define ARITH_BASIC(fn, op, T)                  \
    __device__ T ___##fn(T a, T b)              \
    {                                           \
        return a op b;                          \
    }                                           \


#define ARITH(fn, op)                                   \
    ARITH_BASIC(fn, op, float)                          \
    ARITH_BASIC(fn, op, double)                         \
    ARITH_BASIC(fn, op, int)                            \
    ARITH_BASIC(fn, op, uint)                           \
    ARITH_BASIC(fn, op, char)                           \
    ARITH_BASIC(fn, op, uchar)                          \
    ARITH_BASIC(fn, op, intl)                           \
    ARITH_BASIC(fn, op, uintl)                          \
                                                        \
    __device__ cfloat ___##fn(cfloat a, cfloat b)       \
    {                                                   \
        return cuC##fn##f(a, b);                        \
    }                                                   \
                                                        \
    __device__ cdouble ___##fn(cdouble a, cdouble b)    \
    {                                                   \
        return cuC##fn(a, b);                           \
    }                                                   \

ARITH(add, +)
ARITH(sub, -)
ARITH(mul, *)
ARITH(div, /)
