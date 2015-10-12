/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined(WITH_CPU_LINEAR_ALGEBRA)

#include <inverse.hpp>
#include <err_common.hpp>

#include <af/dim4.hpp>
#include <handle.hpp>
#include <identity.hpp>
#include <copy.hpp>
#include <iostream>
#include <cassert>

#include "lapack_helper.hpp"
#include <cpu_lapack/cpu_lu.hpp>
#include <cpu_lapack/cpu_solve.hpp>

namespace cuda
{
namespace cpu
{

template<typename T>
using getri_func_def = int (*)(ORDER_TYPE, int,
                               T *, int,
                               const int *);

#define INV_FUNC_DEF( FUNC )                                        \
template<typename T> FUNC##_func_def<T> FUNC##_func();

#define INV_FUNC( FUNC, TYPE, PREFIX )                              \
template<> FUNC##_func_def<TYPE>     FUNC##_func<TYPE>()            \
{ return & LAPACK_NAME(PREFIX##FUNC); }

INV_FUNC_DEF( getri )
INV_FUNC(getri , float  , s)
INV_FUNC(getri , double , d)
INV_FUNC(getri , cfloat , c)
INV_FUNC(getri , cdouble, z)

template<typename T>
Array<T> inverse(const Array<T> &in)
{
    int M = in.dims()[0];
    int N = in.dims()[1];

    if (M != N) {
        Array<T> I = identity<T>(in.dims());
        return cpu::solve(in, I);
    }

    Array<T> A = copyArray<T>(in);

    Array<int> pivot = lu_inplace<T>(A, false);

    T *aPtr = pinnedAlloc<T>(A.elements());
    int *pPtr = pinnedAlloc<int>(pivot.elements());
    copyData(aPtr, A);
    copyData(pPtr, pivot);

    getri_func<T>()(AF_LAPACK_COL_MAJOR, M,
                    aPtr, A.strides()[1],
                    pPtr);

    writeHostDataArray<T>(A, aPtr, A.elements() * sizeof(T));

    pinnedFree(aPtr);
    pinnedFree(pPtr);

    return A;
}

#define INSTANTIATE(T)                                                                   \
    template Array<T> inverse<T> (const Array<T> &in);

INSTANTIATE(float)
INSTANTIATE(cfloat)
INSTANTIATE(double)
INSTANTIATE(cdouble)

}
}

#endif
