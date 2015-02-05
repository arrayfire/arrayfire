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


MATH(sinh)
MATH(cosh)
MATH(tanh)

MATH(asinh)
MATH(acosh)
MATH(atanh)
