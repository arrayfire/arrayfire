/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "types.h"

template<typename T> __device__ T sign(T a) { return signbit(a); }

#define MATH_BASIC(fn, T)                       \
    __device__ T ___##fn(T a)                   \
    {                                           \
        return fn(a);                           \
    }                                           \


#define MATH_NOOP(fn, T)                        \
    __device__ T ___##fn(T a)                   \
    {                                           \
        return a;                               \
    }                                           \


#define MATH_CAST(fn, T, Tc)                    \
    __device__ T ___##fn(T a)                   \
    {                                           \
        return (T)fn((Tc)a);                    \
    }                                           \

MATH_BASIC(floor, float)
MATH_BASIC(floor, double)
MATH_NOOP(floor, int)
MATH_NOOP(floor, uint)
MATH_NOOP(floor, char)
MATH_NOOP(floor, uchar)
MATH_NOOP(floor, uintl)
MATH_NOOP(floor, intl)

MATH_BASIC(ceil, float)
MATH_BASIC(ceil, double)
MATH_NOOP(ceil, int)
MATH_NOOP(ceil, uint)
MATH_NOOP(ceil, char)
MATH_NOOP(ceil, uchar)
MATH_NOOP(ceil, uintl)
MATH_NOOP(ceil, intl)

MATH_BASIC(round, float)
MATH_BASIC(round, double)
MATH_NOOP(round, int)
MATH_NOOP(round, uint)
MATH_NOOP(round, char)
MATH_NOOP(round, uchar)
MATH_NOOP(round, uintl)
MATH_NOOP(round, intl)

MATH_BASIC(trunc, float)
MATH_BASIC(trunc, double)
MATH_NOOP(trunc, int)
MATH_NOOP(trunc, uint)
MATH_NOOP(trunc, char)
MATH_NOOP(trunc, uchar)
MATH_NOOP(trunc, uintl)
MATH_NOOP(trunc, intl)

MATH_BASIC(sign, float)
MATH_BASIC(sign, double)
MATH_NOOP(sign, int)
MATH_NOOP(sign, uint)
MATH_NOOP(sign, char)
MATH_NOOP(sign, uchar)
MATH_NOOP(sign, uintl)
MATH_NOOP(sign, intl)

MATH_BASIC(abs, float)
MATH_BASIC(abs, double)
MATH_BASIC(abs, int)
MATH_CAST(abs, char, int)
MATH_NOOP(abs, uint)
MATH_NOOP(abs, uchar)
MATH_NOOP(abs, uintl)
MATH_NOOP(abs, intl)

MATH_BASIC(tgamma, float)
MATH_BASIC(tgamma, double)
MATH_CAST(tgamma, int, float)
MATH_CAST(tgamma, uint, float)
MATH_CAST(tgamma, char, float)
MATH_CAST(tgamma, uchar, float)
MATH_CAST(tgamma, uintl, float)
MATH_CAST(tgamma, intl, float)

MATH_BASIC(lgamma, float)
MATH_BASIC(lgamma, double)
MATH_CAST(lgamma, int, float)
MATH_CAST(lgamma, uint, float)
MATH_CAST(lgamma, char, float)
MATH_CAST(lgamma, uchar, float)
MATH_CAST(lgamma, uintl, float)
MATH_CAST(lgamma, intl, float)

__device__ float ___abs(cfloat a) { return cuCabsf(a); }
__device__ double ___abs(cdouble a) { return cuCabs(a); }

template<typename T> __device__ T rem(T a, T b) { return a % b; }
__device__ float rem(float a, float b) { return remainderf(a, b); }
__device__ double rem(double a, double b) { return remainder(a, b); }

template<typename T> __device__ T mod(T a, T b) { return a % b; }
__device__ float mod(float a, float b) { return fmodf(a, b); }
__device__ double mod(double a, double b) { return fmod(a, b); }

#define MATH2_BASIC(fn, T)                      \
    __device__ T ___##fn(T a, T b)              \
    {                                           \
        return fn(a, b);                        \
    }                                           \

#define MATH2(fn)                                   \
    MATH2_BASIC(fn, float)                          \
    MATH2_BASIC(fn, int)                            \
    MATH2_BASIC(fn, uint)                           \
    MATH2_BASIC(fn, intl)                           \
    MATH2_BASIC(fn, uintl)                          \
    MATH2_BASIC(fn, char)                           \
    MATH2_BASIC(fn, uchar)                          \
    __device__ double ___##fn(double a, double b)   \
    {                                               \
        return fn(a, b);                            \
    }                                               \

MATH2(min)
MATH2(max)
MATH2(mod)
MATH2(rem)

__device__ float ___hypot(float a, float b)
{
    return hypot(a, b);
}

__device__ double ___hypot(double a, double b)
{
    return hypot(a, b);
}

#define COMPARE_CPLX(fn, op, T)                 \
    __device__ T ___##fn(T a, T b)              \
    {                                           \
        return cabs2(a) op cabs2(b) ? a : b;    \
    }                                           \

COMPARE_CPLX(min, <, cfloat)
COMPARE_CPLX(min, <, cdouble)
COMPARE_CPLX(max, >, cfloat)
COMPARE_CPLX(max, >, cdouble)
