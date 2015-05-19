/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "types.h"

#define LOGIC_BASIC(fn, op, T)                  \
    __device__ bool ___##fn(T a, T b)           \
    {                                           \
        return a op b;                          \
    }                                           \


#define LOGIC(fn, op)                               \
    LOGIC_BASIC(fn, op, float)                      \
    LOGIC_BASIC(fn, op, double)                     \
    LOGIC_BASIC(fn, op, int)                        \
    LOGIC_BASIC(fn, op, uint)                       \
    LOGIC_BASIC(fn, op, char)                       \
    LOGIC_BASIC(fn, op, uchar)                      \
    LOGIC_BASIC(fn, op, intl)                       \
    LOGIC_BASIC(fn, op, uintl)                      \
                                                    \
    __device__ bool ___##fn(cfloat a, cfloat b)     \
    {                                               \
        return cabs2(a) op cabs2(b);                \
    }                                               \
                                                    \
    __device__ bool ___##fn(cdouble a, cdouble b)   \
    {                                               \
        return cabs2(a) op cabs2(b);                \
    }                                               \

LOGIC(lt, <)
LOGIC(gt, >)
LOGIC(le, <=)
LOGIC(ge, >=)
LOGIC(and, &&)
LOGIC(or, ||)

#define LOGIC_EQ(fn, op, op2)                       \
    LOGIC_BASIC(fn, op, float)                      \
    LOGIC_BASIC(fn, op, double)                     \
    LOGIC_BASIC(fn, op, int)                        \
    LOGIC_BASIC(fn, op, uint)                       \
    LOGIC_BASIC(fn, op, char)                       \
    LOGIC_BASIC(fn, op, uchar)                      \
                                                    \
    __device__ bool ___##fn(cfloat a, cfloat b)     \
    {                                               \
        return (a.x op b.x) op2 (a.y op b.y);       \
    }                                               \
                                                    \
    __device__ bool ___##fn(cdouble a, cdouble b)   \
    {                                               \
        return (a.x op b.x) op2 (a.y op b.y);       \
    }                                               \

LOGIC_EQ(eq, ==, &&)
LOGIC_EQ(neq, !=, ||)

#define NOT_FN(T)                                   \
    __device__ bool ___not(T in) { return !in; }    \

NOT_FN(float)
NOT_FN(double)
NOT_FN(int)
NOT_FN(uint)
NOT_FN(char)
NOT_FN(uchar)
NOT_FN(intl)
NOT_FN(uintl)

#define BIT_FN(T)                                                   \
    __device__ T ___bitand   (T lhs, T rhs) { return lhs &  rhs; }  \
    __device__ T ___bitor    (T lhs, T rhs) { return lhs |  rhs; }  \
    __device__ T ___bitxor   (T lhs, T rhs) { return lhs ^  rhs; }  \
    __device__ T ___bitshiftl(T lhs, T rhs) { return lhs << rhs; }  \
    __device__ T ___bitshiftr(T lhs, T rhs) { return lhs >> rhs; }  \

BIT_FN(int)
BIT_FN(char)
BIT_FN(intl)
BIT_FN(uchar)
BIT_FN(uint)
BIT_FN(uintl)

__device__ char ___isNaN(float in) { return isnan(in); }
__device__ char ___isINF(float in) { return isinf(in); }
__device__ char ___iszero(float in) { return (in == 0); }

__device__ char ___isNaN(double in) { return isnan(in); }
__device__ char ___isINF(double in) { return isinf(in); }
__device__ char ___iszero(double in) { return (in == 0); }
