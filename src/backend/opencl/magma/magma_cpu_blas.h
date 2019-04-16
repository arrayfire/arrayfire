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
#include <common/blas_headers.hpp>
#include <common/defines.hpp>
#include <err_opencl.hpp>
#include "magma_types.h"

#define CPU_BLAS_FUNC_DEF(NAME) \
    template<typename T>        \
    struct cpu_blas_##NAME##_func;

#define CPU_BLAS_FUNC1(NAME, TYPE, X)                \
    template<>                                       \
    struct cpu_blas_##NAME##_func<TYPE> {            \
        template<typename... Args>                   \
        void operator()(Args... args) {              \
            cblas_##X##NAME(CblasColMajor, args...); \
        }                                            \
    };

#define CPU_BLAS_FUNC2(NAME, TYPE, X)     \
    template<>                            \
    struct cpu_blas_##NAME##_func<TYPE> { \
        template<typename... Args>        \
        void operator()(Args... args) {   \
            cblas_##X##NAME(args...);     \
        }                                 \
    };

#define CPU_BLAS_DECL1(NAME)                   \
    CPU_BLAS_FUNC_DEF(NAME)                    \
    CPU_BLAS_FUNC1(NAME, float, s)             \
    CPU_BLAS_FUNC1(NAME, double, d)            \
    CPU_BLAS_FUNC1(NAME, magmaFloatComplex, c) \
    CPU_BLAS_FUNC1(NAME, magmaDoubleComplex, z)

#define CPU_BLAS_DECL2(NAME)                   \
    CPU_BLAS_FUNC_DEF(NAME)                    \
    CPU_BLAS_FUNC2(NAME, float, s)             \
    CPU_BLAS_FUNC2(NAME, double, d)            \
    CPU_BLAS_FUNC2(NAME, magmaFloatComplex, c) \
    CPU_BLAS_FUNC2(NAME, magmaDoubleComplex, z)

CPU_BLAS_DECL1(gemv)
CPU_BLAS_DECL2(scal)
CPU_BLAS_DECL2(axpy)

inline float *cblas_ptr(float *in) { return in; }
inline double *cblas_ptr(double *in) { return in; }

#if defined(IS_OPENBLAS)
inline float *cblas_ptr(magmaFloatComplex *in) { return (float *)in; }
inline double *cblas_ptr(magmaDoubleComplex *in) { return (double *)in; }
#else
inline void *cblas_ptr(magmaFloatComplex *in) { return (void *)in; }
inline void *cblas_ptr(magmaDoubleComplex *in) { return (void *)in; }
#endif

inline float cblas_scalar(float *in) { return *in; }
inline double cblas_scalar(double *in) { return *in; }

#if defined(IS_OPENBLAS)
inline float *cblas_scalar(magmaFloatComplex *in) { return (float *)in; }
inline double *cblas_scalar(magmaDoubleComplex *in) { return (double *)in; }
#else
inline void *cblas_scalar(magmaFloatComplex *in) { return (void *)in; }
inline void *cblas_scalar(magmaDoubleComplex *in) { return (void *)in; }
#endif

#endif
