/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#ifndef MAGMA_CPU_LAPACK
#define MAGMA_CPU_LAPACK

#include "magma_types.h"

#define LAPACKE_sunmqr_work(...) LAPACKE_sormqr_work(__VA_ARGS__)
#define LAPACKE_dunmqr_work(...) LAPACKE_dormqr_work(__VA_ARGS__)
#define LAPACKE_sungqr_work(...) LAPACKE_sorgqr_work(__VA_ARGS__)
#define LAPACKE_dungqr_work(...) LAPACKE_dorgqr_work(__VA_ARGS__)

#define lapack_complex_float magmaFloatComplex
#define lapack_complex_double magmaDoubleComplex
#define LAPACK_PREFIX LAPACKE_
#define ORDER_TYPE int
#define LAPACK_NAME(fn) LAPACKE_##fn

#if defined(__APPLE__)
    #define LAPACK_COL_MAJOR 102
    #include "../../lapacke.hpp"
#else
    #ifdef USE_MKL
        #include<mkl_lapacke.h>
    #else // NETLIB LAPACKE
        #include<lapacke.h>
    #endif  // MKL/NETLIB
#endif  //APPLE

#define CPU_LAPACK_FUNC_DEF(NAME)               \
    template<typename T>                        \
    struct NAME##_func;

#define CPU_LAPACK_FUNC(NAME, TYPE, X)              \
    template<>                                      \
    struct NAME##_func<TYPE>                        \
    {                                               \
        template<typename... Args>                  \
            int                                     \
            operator() (Args... args)               \
        { return LAPACK_NAME(X##NAME)(args...); }   \
    };

CPU_LAPACK_FUNC_DEF(getrf)
CPU_LAPACK_FUNC(getrf, float,      s)
CPU_LAPACK_FUNC(getrf, double,     d)
CPU_LAPACK_FUNC(getrf, magmaFloatComplex,     c)
CPU_LAPACK_FUNC(getrf, magmaDoubleComplex,    z)

CPU_LAPACK_FUNC_DEF(potrf)
CPU_LAPACK_FUNC(potrf, float,      s)
CPU_LAPACK_FUNC(potrf, double,     d)
CPU_LAPACK_FUNC(potrf, magmaFloatComplex,     c)
CPU_LAPACK_FUNC(potrf, magmaDoubleComplex,    z)

CPU_LAPACK_FUNC_DEF(trtri)
CPU_LAPACK_FUNC(trtri, float,      s)
CPU_LAPACK_FUNC(trtri, double,     d)
CPU_LAPACK_FUNC(trtri, magmaFloatComplex,     c)
CPU_LAPACK_FUNC(trtri, magmaDoubleComplex,    z)

CPU_LAPACK_FUNC_DEF(geqrf_work)
CPU_LAPACK_FUNC(geqrf_work, float,      s)
CPU_LAPACK_FUNC(geqrf_work, double,     d)
CPU_LAPACK_FUNC(geqrf_work, magmaFloatComplex,     c)
CPU_LAPACK_FUNC(geqrf_work, magmaDoubleComplex,    z)

CPU_LAPACK_FUNC_DEF(larft)
CPU_LAPACK_FUNC(larft, float,      s)
CPU_LAPACK_FUNC(larft, double,     d)
CPU_LAPACK_FUNC(larft, magmaFloatComplex,     c)
CPU_LAPACK_FUNC(larft, magmaDoubleComplex,    z)

CPU_LAPACK_FUNC_DEF(unmqr_work)
CPU_LAPACK_FUNC(unmqr_work, float,      s)
CPU_LAPACK_FUNC(unmqr_work, double,     d)
CPU_LAPACK_FUNC(unmqr_work, magmaFloatComplex,     c)
CPU_LAPACK_FUNC(unmqr_work, magmaDoubleComplex,    z)

CPU_LAPACK_FUNC_DEF(ungqr_work)
CPU_LAPACK_FUNC(ungqr_work, float,      s)
CPU_LAPACK_FUNC(ungqr_work, double,     d)
CPU_LAPACK_FUNC(ungqr_work, magmaFloatComplex,     c)
CPU_LAPACK_FUNC(ungqr_work, magmaDoubleComplex,    z)

CPU_LAPACK_FUNC_DEF(laswp)
CPU_LAPACK_FUNC(laswp, float,      s)
CPU_LAPACK_FUNC(laswp, double,     d)
CPU_LAPACK_FUNC(laswp, magmaFloatComplex,     c)
CPU_LAPACK_FUNC(laswp, magmaDoubleComplex,    z)

#endif
