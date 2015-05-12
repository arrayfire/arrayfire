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
#include <err_clblas.hpp>

using opencl::cfloat;
using opencl::cdouble;

#define BLAS_FUNC_DEF(NAME)                     \
    template<typename T>                        \
    struct NAME##_func;

#define BLAS_FUNC(NAME, TYPE, PREFIX)                       \
    template<>                                              \
    struct NAME##_func<TYPE>                                \
    {                                                       \
        template<typename... Args>                          \
            void                                            \
            operator() (Args... args)                       \
        {                                                   \
            CLBLAS_CHECK(clblas##PREFIX##NAME(args...));    \
        }                                                   \
    };

BLAS_FUNC_DEF(gemm)
BLAS_FUNC(gemm, float,      S)
BLAS_FUNC(gemm, double,     D)
BLAS_FUNC(gemm, cfloat,     C)
BLAS_FUNC(gemm, cdouble,    Z)

BLAS_FUNC_DEF(trmm)
BLAS_FUNC(trmm, float,      S)
BLAS_FUNC(trmm, double,     D)
BLAS_FUNC(trmm, cfloat,     C)
BLAS_FUNC(trmm, cdouble,    Z)

BLAS_FUNC_DEF(trsm)
BLAS_FUNC(trsm, float,      S)
BLAS_FUNC(trsm, double,     D)
BLAS_FUNC(trsm, cfloat,     C)
BLAS_FUNC(trsm, cdouble,    Z)

BLAS_FUNC_DEF(trsv)
BLAS_FUNC(trsv, float,      S)
BLAS_FUNC(trsv, double,     D)
BLAS_FUNC(trsv, cfloat,     C)
BLAS_FUNC(trsv, cdouble,    Z)

#define clblasSherk(...) clblasSsyrk(__VA_ARGS__)
#define clblasDherk(...) clblasDsyrk(__VA_ARGS__)

BLAS_FUNC_DEF(herk)
BLAS_FUNC(herk, float,      S)
BLAS_FUNC(herk, double,     D)
BLAS_FUNC(herk, cfloat,     C)
BLAS_FUNC(herk, cdouble,    Z)

#endif
