/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#ifndef __MAGMA_BLAS_H
#define __MAGMA_BLAS_H

#include "magma_common.h"
#include <types.hpp>
#include <clBLAS.h>

using opencl::cfloat;
using opencl::cdouble;

#define BLAS_FUNC_DEF(NAME)                                             \
template<typename T>                                                    \
struct NAME##_func;

#define BLAS_FUNC(NAME, TYPE, PREFIX)                                   \
template<>                                                              \
struct NAME##_func<TYPE>                                                \
{                                                                       \
    template<typename... Args>                                          \
    clblasStatus                                                        \
    operator() (Args... args) { return clblas##PREFIX##NAME(args...); } \
};

BLAS_FUNC_DEF(gemm)
BLAS_FUNC(gemm, float,      S)
BLAS_FUNC(gemm, double,     D)
BLAS_FUNC(gemm, cfloat,     C)
BLAS_FUNC(gemm, cdouble,    Z)

BLAS_FUNC_DEF(gemv)
BLAS_FUNC(gemv, float,      S)
BLAS_FUNC(gemv, double,     D)
BLAS_FUNC(gemv, cfloat,     C)
BLAS_FUNC(gemv, cdouble,    Z)

BLAS_FUNC_DEF(trsm)
BLAS_FUNC(trsm, float,      S)
BLAS_FUNC(trsm, double,     D)
BLAS_FUNC(trsm, cfloat,     C)
BLAS_FUNC(trsm, cdouble,    Z)

#endif
