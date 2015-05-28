/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <solve.hpp>
#include <err_common.hpp>

#if defined(WITH_CPU_LINEAR_ALGEBRA)

#include <af/dim4.hpp>
#include <handle.hpp>
#include <range.hpp>
#include <iostream>
#include <cassert>
#include <err_cpu.hpp>

#include <lapack_helper.hpp>

namespace cpu
{

template<typename T>
using gesv_func_def = int (*)(ORDER_TYPE, int, int,
                              T *, int,
                              int *,
                              T *, int);

template<typename T>
using gels_func_def = int (*)(ORDER_TYPE, char,
                              int, int, int,
                              T *, int,
                              T *, int);

template<typename T>
using getrs_func_def = int (*)(ORDER_TYPE, char,
                               int, int,
                               const T *, int,
                               const int *,
                               T *, int);

template<typename T>
using trtrs_func_def = int (*)(ORDER_TYPE,
                               char, char, char,
                               int, int,
                               const T *, int,
                               T *, int);


#define SOLVE_FUNC_DEF( FUNC )                                      \
template<typename T> FUNC##_func_def<T> FUNC##_func();


#define SOLVE_FUNC( FUNC, TYPE, PREFIX )                            \
template<> FUNC##_func_def<TYPE>     FUNC##_func<TYPE>()            \
{ return & LAPACK_NAME(PREFIX##FUNC); }

SOLVE_FUNC_DEF( gesv )
SOLVE_FUNC(gesv , float  , s)
SOLVE_FUNC(gesv , double , d)
SOLVE_FUNC(gesv , cfloat , c)
SOLVE_FUNC(gesv , cdouble, z)

SOLVE_FUNC_DEF( gels )
SOLVE_FUNC(gels , float  , s)
SOLVE_FUNC(gels , double , d)
SOLVE_FUNC(gels , cfloat , c)
SOLVE_FUNC(gels , cdouble, z)

SOLVE_FUNC_DEF( getrs )
SOLVE_FUNC(getrs , float  , s)
SOLVE_FUNC(getrs , double , d)
SOLVE_FUNC(getrs , cfloat , c)
SOLVE_FUNC(getrs , cdouble, z)

SOLVE_FUNC_DEF( trtrs )
SOLVE_FUNC(trtrs , float  , s)
SOLVE_FUNC(trtrs , double , d)
SOLVE_FUNC(trtrs , cfloat , c)
SOLVE_FUNC(trtrs , cdouble, z)

template<typename T>
Array<T> solveLU(const Array<T> &A, const Array<int> &pivot,
                 const Array<T> &b, const af_mat_prop options)
{
    int N = A.dims()[0];
    int NRHS = b.dims()[1];

    Array< T > B = copyArray<T>(b);

    getrs_func<T>()(AF_LAPACK_COL_MAJOR, 'N',
                    N, NRHS,
                    A.get(), A.strides()[1],
                    pivot.get(),
                    B.get(), B.strides()[1]);

    return B;
}

template<typename T>
Array<T> triangleSolve(const Array<T> &A, const Array<T> &b, const af_mat_prop options)
{
    Array<T> B = copyArray<T>(b);
    int N = B.dims()[0];
    int NRHS = B.dims()[1];

    trtrs_func<T>()(AF_LAPACK_COL_MAJOR,
                    options & AF_MAT_UPPER ? 'U' : 'L',
                    'N', // transpose flag
                    options & AF_MAT_DIAG_UNIT ? 'U' : 'N',
                    N, NRHS,
                    A.get(), A.strides()[1],
                    B.get(), B.strides()[1]);
    return B;
}


template<typename T>
Array<T> solve(const Array<T> &a, const Array<T> &b, const af_mat_prop options)
{

    if (options & AF_MAT_UPPER ||
        options & AF_MAT_LOWER) {
        return triangleSolve<T>(a, b, options);
    }

    int M = a.dims()[0];
    int N = a.dims()[1];
    int K = b.dims()[1];


    Array<T> A = copyArray<T>(a);
    Array<T> B = padArray<T, T>(b, dim4(max(M, N), K));

    if(M == N) {
        Array<int> pivot = createEmptyArray<int>(dim4(N, 1, 1));
        gesv_func<T>()(AF_LAPACK_COL_MAJOR, N, K,
                       A.get(), A.strides()[1],
                       pivot.get(),
                       B.get(), B.strides()[1]);
    } else {
        int sM = a.strides()[1];
        int sN = a.strides()[2] / sM;

        gels_func<T>()(AF_LAPACK_COL_MAJOR, 'N',
                       M, N, K,
                       A.get(), A.strides()[1],
                       B.get(), max(sM, sN));
        B.resetDims(dim4(N, K));
    }

    return B;
}

#define INSTANTIATE_SOLVE(T)                                            \
    template Array<T> solve<T>(const Array<T> &a, const Array<T> &b,    \
                               const af_mat_prop options);              \
    template Array<T> solveLU<T>(const Array<T> &A, const Array<int> &pivot, \
                                 const Array<T> &b, const af_mat_prop options); \

INSTANTIATE_SOLVE(float)
INSTANTIATE_SOLVE(cfloat)
INSTANTIATE_SOLVE(double)
INSTANTIATE_SOLVE(cdouble)

}

#else

namespace cpu
{

template<typename T>
Array<T> solveLU(const Array<T> &A, const Array<int> &pivot,
                 const Array<T> &b, const af_mat_prop options)
{
    AF_ERROR("Linear Algebra is diabled on CPU",
             AF_ERR_NOT_CONFIGURED);
}

template<typename T>
Array<T> solve(const Array<T> &a, const Array<T> &b, const af_mat_prop options)
{
    AF_ERROR("Linear Algebra is diabled on CPU",
              AF_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE_SOLVE(T)                                            \
    template Array<T> solve<T>(const Array<T> &a, const Array<T> &b,    \
                               const af_mat_prop options);              \
    template Array<T> solveLU<T>(const Array<T> &A, const Array<int> &pivot, \
                                 const Array<T> &b, const af_mat_prop options); \

INSTANTIATE_SOLVE(float)
INSTANTIATE_SOLVE(cfloat)
INSTANTIATE_SOLVE(double)
INSTANTIATE_SOLVE(cdouble)
}

#endif
