/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#ifndef AF_OPENCL_CPU
#define AF_OPENCL_CPU

#include <af/defines.h>
#include <Array.hpp>
#include <memory.hpp>
#include <types.hpp>
#include <err_common.hpp>
#include <platform.hpp>

//********************************************************/
// LAPACK
//********************************************************/
#if defined(WITH_OPENCL_LINEAR_ALGEBRA)

#define lapack_complex_float opencl::cfloat
#define lapack_complex_double opencl::cdouble
#define LAPACK_PREFIX LAPACKE_
#define ORDER_TYPE int
#define AF_LAPACK_COL_MAJOR LAPACK_COL_MAJOR
#define LAPACK_NAME(fn) LAPACKE_##fn

#ifdef USE_MKL
    #include<mkl_lapacke.h>
#else
    #ifdef __APPLE__
        #include <Accelerate/Accelerate.h>
        #include <lapacke.hpp>
        #undef AF_LAPACK_COL_MAJOR
        #define AF_LAPACK_COL_MAJOR 0
    #else // NETLIB LAPACKE
        #include<lapacke.h>
    #endif
#endif

#endif // WITH_OPENCL_LINEAR_ALGEBRA

//********************************************************/
// BLAS
//********************************************************/
#ifdef USE_MKL
    #include <mkl_cblas.h>
#else
    #ifdef __APPLE__
        #include <Accelerate/Accelerate.h>
    #else
        extern "C" {
            #include <cblas.h>
        }
    #endif
#endif

// TODO: Ask upstream for a more official way to detect it
#ifdef OPENBLAS_CONST
#define IS_OPENBLAS
#endif

// Make sure we get the correct type signature for OpenBLAS
// OpenBLAS defines blasint as it's index type. Emulate this
// if we're not dealing with openblas and use it where applicable
#ifndef IS_OPENBLAS
typedef int blasint;
#endif

#endif
