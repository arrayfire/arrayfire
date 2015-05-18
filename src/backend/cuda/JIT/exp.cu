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


MATH(exp)
MATH(expm1)
MATH(erf)
MATH(erfc)

MATH(log)
MATH(log10)
MATH(log1p)
MATH(log2)

MATH(sqrt)
MATH(cbrt)

#define MATH2_BASIC(fn, T)                      \
    __device__ T ___##fn(T a, T b)              \
    {                                           \
        return fn##f((float)a, (float)b);       \
    }                                           \

#define MATH2(fn)                                   \
    MATH2_BASIC(fn, float)                          \
    MATH2_BASIC(fn, int)                            \
    MATH2_BASIC(fn, uint)                           \
    MATH2_BASIC(fn, char)                           \
    MATH2_BASIC(fn, uchar)                          \
    __device__ double ___##fn(double a, double b)   \
    {                                               \
        return fn(a, b);                            \
    }                                               \

MATH2(pow)

__device__ cfloat ___pow(cfloat a, float b)
{
    float R = cuCabsf(a);
    float Theta = atan2(a.y, a.x);
    float R_b = powf(R, b);
    float Theta_b = Theta * b;
    cfloat res = {R_b * cosf(Theta_b), R_b * sinf(Theta_b)};
    return res;
}

__device__ cdouble ___pow(cdouble a, float b)
{
    float R = cuCabs(a);
    float Theta = atan2(a.y, a.x);
    float R_b = pow(R, b);
    float Theta_b = Theta * b;
    cdouble res = {R_b * cos(Theta_b), R_b * sin(Theta_b)};
    return res;
}
