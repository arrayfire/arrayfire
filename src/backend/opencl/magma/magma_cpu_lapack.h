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

#define lapack_complex_float magmaFloatComplex
#define lapack_complex_double magmaDoubleComplex
#define LAPACK_PREFIX LAPACKE_
#define ORDER_TYPE int
#define LAPACK_NAME(fn) LAPACKE_##fn
#include<lapacke.h>

#define CPU_LAPACK_FUNC_DEF(NAME)               \
    template<typename T>                        \
    struct NAME##_func;

#define CPU_LAPACK_FUNC(NAME, TYPE, X)          \
    template<>                                  \
    struct NAME##_func<TYPE>                    \
    {                                           \
        template<typename... Args>              \
            void                                \
            operator() (Args... args)           \
        { (LAPACK_NAME(X##NAME))(args...); }    \
    };

CPU_LAPACK_FUNC_DEF(getrf)
CPU_LAPACK_FUNC(getrf, float,      s)
CPU_LAPACK_FUNC(getrf, double,     d)
CPU_LAPACK_FUNC(getrf, magmaFloatComplex,     c)
CPU_LAPACK_FUNC(getrf, magmaDoubleComplex,    z)

#endif
