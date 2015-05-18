/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "types.h"

#define CAST_BASIC(FN, To, Ti) __device__ To FN(Ti in) { return (To) in; }

#define CAST_BASIC_BOOL(FN, To, Ti) __device__ To FN(Ti in) { return (To)(in != 0); }

#define CAST(T, X)                              \
    CAST_BASIC(___mk##X, T, float)              \
    CAST_BASIC(___mk##X, T, double)             \
    CAST_BASIC(___mk##X, T, int)                \
    CAST_BASIC(___mk##X, T, uint)               \
    CAST_BASIC(___mk##X, T, char)               \
    CAST_BASIC(___mk##X, T, uchar)              \
    CAST_BASIC(___mk##X, T, intl)               \
    CAST_BASIC(___mk##X, T, uintl)              \

CAST(float, S)
CAST(double, D)
CAST(int, I)
CAST(intl, X)
CAST(uint, U)
CAST(uchar, V)
CAST(uintl, Y)

CAST_BASIC_BOOL(___mkJ, char, float)
CAST_BASIC_BOOL(___mkJ, char, double)
CAST_BASIC_BOOL(___mkJ, char, int)
CAST_BASIC_BOOL(___mkJ, char, uint)
CAST_BASIC_BOOL(___mkJ, char, char)
CAST_BASIC_BOOL(___mkJ, char, uchar)
CAST_BASIC_BOOL(___mkJ, char, intl)
CAST_BASIC_BOOL(___mkJ, char, uintl)

#define CPLX_BASIC(FN, To, Tr, Ti)              \
    __device__ To FN(Ti in)                     \
    {                                           \
        To out = {(Tr)in, 0};                   \
        return out;                             \
    }                                           \

#define CPLX_CAST(T, Tr, X)                     \
    CPLX_BASIC(___mk##X, T, Tr, float)          \
    CPLX_BASIC(___mk##X, T, Tr, double)         \
    CPLX_BASIC(___mk##X, T, Tr, int)            \
    CPLX_BASIC(___mk##X, T, Tr, uint)           \
    CPLX_BASIC(___mk##X, T, Tr, char)           \
    CPLX_BASIC(___mk##X, T, Tr, uchar)          \

CPLX_CAST(cfloat, float, C)
CPLX_CAST(cdouble, double, Z)

__device__ cfloat ___mkC(cfloat C)
{
    return C;
}

__device__ cfloat ___mkC(cdouble C)
{
    cfloat res = {C.x, C.y};
    return res;
}

__device__ cdouble ___mkZ(cdouble C)
{
    return C;
}

__device__ cdouble ___mkZ(cfloat C)
{
    cdouble res = {C.x, C.y};
    return res;
}

__device__ float ___real(cfloat in) { return in.x; }
__device__ double ___real(cdouble in) { return in.x; }


__device__ float ___imag(cfloat in) { return in.y; }
__device__ double ___imag(cdouble in) { return in.y; }

__device__ cfloat ___cplx(float l, float r) { cfloat out = {l, r}; return out; }
__device__ cdouble ___cplx(double l, double r) { cdouble out = {l, r}; return out; }

__device__ cfloat  ___conj(cfloat  in) { return cuConjf(in); }
__device__ cdouble ___conj(cdouble in) { return cuConj (in); }
