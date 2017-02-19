/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <defines.hpp>

#include <clBLAS.h>
#include <err_clblas.hpp>
#include <mutex>  // for std::once_flag

// Convert MAGMA constants to clBLAS constants
clblasOrder          clblas_order_const( magma_order_t order );
clblasTranspose      clblas_trans_const( magma_trans_t trans );
clblasUplo           clblas_uplo_const ( magma_uplo_t  uplo  );
clblasDiag           clblas_diag_const ( magma_diag_t  diag  );
clblasSide           clblas_side_const ( magma_side_t  side  );

// Error checking
#define OPENCL_BLAS_CHECK CLBLAS_CHECK

// Transposing
#define OPENCL_BLAS_TRANS_T clblasTranspose // the type
#define OPENCL_BLAS_NO_TRANS clblasNoTrans
#define OPENCL_BLAS_TRANS clblasTrans
#define OPENCL_BLAS_CONJ_TRANS clblasConjTrans

// Triangles
#define OPENCL_BLAS_TRIANGLE_T clblasUplo // the type
#define OPENCL_BLAS_TRIANGLE_UPPER clblasUpper
#define OPENCL_BLAS_TRIANGLE_LOWER clblasLower

// Sides
#define OPENCL_BLAS_SIDE_RIGHT clblasRight
#define OPENCL_BLAS_SIDE_LEFT clblasLeft

// Unit or non-unit diagonal
#define OPENCL_BLAS_UNIT_DIAGONAL clblasUnit
#define OPENCL_BLAS_NON_UNIT_DIAGONAL clblasNonUnit

// Initialization of the OpenCL BLAS library
inline void gpu_blas_init()
{
    static std::once_flag clblasSetupFlag;
    call_once(clblasSetupFlag, clblasSetup);
}

#define clblasSherk(...) clblasSsyrk(__VA_ARGS__)
#define clblasDherk(...) clblasDsyrk(__VA_ARGS__)

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
