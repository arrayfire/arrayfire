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
#include <defines.hpp>
#include <types.hpp>
#include <clBLAS.h>
#include <err_clblas.hpp>

using opencl::cfloat;
using opencl::cdouble;

#define clblasSherk(...) clblasSsyrk(__VA_ARGS__)
#define clblasDherk(...) clblasDsyrk(__VA_ARGS__)

#define BLAS_FUNC_DEF(NAME)                     \
    template<typename T>                        \
    struct gpu_blas_##NAME##_func;

#define BLAS_FUNC(NAME, TYPE, PREFIX)                       \
    template<>                                              \
    struct gpu_blas_##NAME##_func<TYPE>                     \
    {                                                       \
        template<typename... Args>                          \
            clblasStatus                                    \
            operator() (Args... args)                       \
        {                                                   \
            return clblas##PREFIX##NAME(clblasColumnMajor,  \
                                        args...);           \
        }                                                   \
    };

#define BLAS_FUNC_DECL(NAME)                    \
    BLAS_FUNC_DEF(NAME)                         \
    BLAS_FUNC(NAME, float,      S)              \
    BLAS_FUNC(NAME, double,     D)              \
    BLAS_FUNC(NAME, cfloat,     C)              \
    BLAS_FUNC(NAME, cdouble,    Z)              \

BLAS_FUNC_DECL(gemm)
BLAS_FUNC_DECL(gemv)
BLAS_FUNC_DECL(trmm)
BLAS_FUNC_DECL(trsm)
BLAS_FUNC_DECL(trsv)
BLAS_FUNC_DECL(herk)

#endif
