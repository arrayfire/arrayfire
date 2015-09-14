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

#include <err_common.hpp>
#include <defines.hpp>
#include "magma_types.h"

#define LAPACKE_sunmqr_work(...) LAPACKE_sormqr_work(__VA_ARGS__)
#define LAPACKE_dunmqr_work(...) LAPACKE_dormqr_work(__VA_ARGS__)
#define LAPACKE_sungqr_work(...) LAPACKE_sorgqr_work(__VA_ARGS__)
#define LAPACKE_dungqr_work(...) LAPACKE_dorgqr_work(__VA_ARGS__)
#define LAPACKE_sungbr_work(...) LAPACKE_sorgbr_work(__VA_ARGS__)
#define LAPACKE_dungbr_work(...) LAPACKE_dorgbr_work(__VA_ARGS__)

template<typename... Args>
int LAPACKE_slacgv(Args... args) { return 0; }

template<typename... Args>
int LAPACKE_dlacgv(Args... args) { return 0; }

template<typename... Args>
int LAPACKE_slacgv_work(Args... args) { return 0; }

template<typename... Args>
int LAPACKE_dlacgv_work(Args... args) { return 0; }

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

#define LAPACKE_CHECK(fn) do {                  \
        int __info = fn;                        \
        if (__info != 0) {                      \
            char lapacke_st_msg[32];            \
            snprintf(lapacke_st_msg,            \
                     sizeof(lapacke_st_msg),    \
                     "LAPACKE Error (%d)",      \
                     (int)(__info));            \
            AF_ERROR(lapacke_st_msg,            \
                     AF_ERR_INTERNAL);          \
        }                                       \
    } while(0)

#define CPU_LAPACK_FUNC_DEF(NAME)               \
    template<typename T>                        \
    struct cpu_lapack_##NAME##_func;

#define CPU_LAPACK_FUNC1(NAME, TYPE, X)                     \
    template<>                                              \
    struct cpu_lapack_##NAME##_func<TYPE>                   \
    {                                                       \
        template<typename... Args>                          \
            int                                             \
            operator() (Args... args)                       \
        {                                                   \
            return LAPACK_NAME(X##NAME)(LAPACK_COL_MAJOR,   \
                                        args...);           \
        }                                                   \
    };

#define CPU_LAPACK_FUNC2(NAME, TYPE, X)             \
    template<>                                      \
    struct cpu_lapack_##NAME##_func<TYPE>           \
    {                                               \
        template<typename... Args>                  \
            int                                     \
            operator() (Args... args)               \
        {                                           \
            return LAPACK_NAME(X##NAME)(args...);   \
        }                                           \
    };

#define CPU_LAPACK_FUNC3(NAME, TYPE, X)             \
    template<>                                      \
    struct cpu_lapack_##NAME##_func<TYPE>           \
    {                                               \
        template<typename... Args>                  \
            double                                  \
            operator() (Args... args)               \
        { return LAPACK_NAME(X##NAME)(args...); }   \
    };

#define CPU_LAPACK_DECL1(NAME)                          \
    CPU_LAPACK_FUNC_DEF(NAME)                           \
    CPU_LAPACK_FUNC1(NAME, float,      s)               \
    CPU_LAPACK_FUNC1(NAME, double,     d)               \
    CPU_LAPACK_FUNC1(NAME, magmaFloatComplex,     c)    \
    CPU_LAPACK_FUNC1(NAME, magmaDoubleComplex,    z)    \

#define CPU_LAPACK_DECL2(NAME)                          \
    CPU_LAPACK_FUNC_DEF(NAME)                           \
    CPU_LAPACK_FUNC2(NAME, float,      s)               \
    CPU_LAPACK_FUNC2(NAME, double,     d)               \
    CPU_LAPACK_FUNC2(NAME, magmaFloatComplex,     c)    \
    CPU_LAPACK_FUNC2(NAME, magmaDoubleComplex,    z)    \

#define CPU_LAPACK_DECL3(NAME)                  \
    CPU_LAPACK_FUNC_DEF(NAME)                   \
    CPU_LAPACK_FUNC3(NAME, float,      s)       \
    CPU_LAPACK_FUNC3(NAME, double,     d)       \

CPU_LAPACK_DECL1(getrf)
CPU_LAPACK_DECL1(gebrd_work)
CPU_LAPACK_DECL1(potrf)
CPU_LAPACK_DECL1(trtri)
CPU_LAPACK_DECL1(geqrf_work)
CPU_LAPACK_DECL1(larft)
CPU_LAPACK_DECL1(unmqr_work)
CPU_LAPACK_DECL1(ungqr_work)
CPU_LAPACK_DECL1(ungbr_work)
CPU_LAPACK_DECL1(bdsqr_work)
CPU_LAPACK_DECL1(laswp)
CPU_LAPACK_DECL1(laset)

CPU_LAPACK_DECL2(lacgv_work)
CPU_LAPACK_DECL2(larfg_work)
CPU_LAPACK_DECL1(lacpy)
CPU_LAPACK_DECL3(lamch)

#endif
