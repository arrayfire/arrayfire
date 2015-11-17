/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#ifndef MAGMA_CPU_BLAS
#define MAGMA_CPU_BLAS
#include <err_common.hpp>
#include <defines.hpp>
#include "magma_types.h"

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#ifdef USE_MKL
#include <mkl_cblas.h>
#else
extern "C" {
#include <cblas.h>
}
#endif
#endif

// Todo: Ask upstream for a more official way to detect it
#ifdef OPENBLAS_CONST
#define IS_OPENBLAS
#endif

// Make sure we get the correct type signature for OpenBLAS
// OpenBLAS defines blasint as it's index type. Emulate this
// if we're not dealing with openblas and use it where applicable
#ifndef IS_OPENBLAS
typedef int blasint;
#endif

#define CPU_BLAS_FUNC_DEF(NAME)                 \
    template<typename T>                        \
    struct cpu_blas_##NAME##_func;

#define CPU_BLAS_FUNC1(NAME, TYPE, X)                       \
    template<>                                              \
    struct cpu_blas_##NAME##_func<TYPE>                     \
    {                                                       \
        template<typename... Args>                          \
            void                                            \
            operator() (Args... args)                       \
        { return cblas_##X##NAME(CblasColMajor, args...); } \
    };

#define CPU_BLAS_FUNC2(NAME, TYPE, X)           \
    template<>                                  \
    struct cpu_blas_##NAME##_func<TYPE>         \
    {                                           \
        template<typename... Args>              \
            void                                \
            operator() (Args... args)           \
        { return cblas_##X##NAME(args...); }    \
    };

#define CPU_BLAS_DECL1(NAME)                        \
    CPU_BLAS_FUNC_DEF(NAME)                         \
    CPU_BLAS_FUNC1(NAME, float,      s)             \
    CPU_BLAS_FUNC1(NAME, double,     d)             \
    CPU_BLAS_FUNC1(NAME, magmaFloatComplex,     c)  \
    CPU_BLAS_FUNC1(NAME, magmaDoubleComplex,    z)  \

#define CPU_BLAS_DECL2(NAME)                        \
    CPU_BLAS_FUNC_DEF(NAME)                         \
    CPU_BLAS_FUNC2(NAME, float,      s)             \
    CPU_BLAS_FUNC2(NAME, double,     d)             \
    CPU_BLAS_FUNC2(NAME, magmaFloatComplex,     c)  \
    CPU_BLAS_FUNC2(NAME, magmaDoubleComplex,    z)  \

CPU_BLAS_DECL1(gemv)
CPU_BLAS_DECL2(scal)
CPU_BLAS_DECL2(axpy)

inline float * cblas_ptr(float *in) { return in; }
inline double * cblas_ptr(double *in) { return in; }
inline void * cblas_ptr(magmaFloatComplex *in) { return (void *)in; }
inline void * cblas_ptr(magmaDoubleComplex *in) { return (void *)in; }

inline float cblas_scalar(float *in) { return *in; }
inline double cblas_scalar(double *in) { return *in; }
inline void *cblas_scalar(magmaFloatComplex *in) { return (void *)in; }
inline void *cblas_scalar(magmaDoubleComplex *in) { return (void *)in; }
#endif
