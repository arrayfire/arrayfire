/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <err_common.hpp>
#include <solve.hpp>

#if defined(WITH_LINEAR_ALGEBRA)

#include <cusolverDnManager.hpp>
#include <cublas_v2.h>
#include <identity.hpp>
#include <iostream>
#include <memory.hpp>
#include <copy.hpp>
#include <transpose.hpp>

#include <math.hpp>
#include <err_common.hpp>

#include <blas.hpp>
#include <lu.hpp>
#include <qr.hpp>

#include <handle.hpp>
#include <cstdio>

namespace cuda
{

using cusolver::getDnHandle;

//cusolverStatus_t cusolverDn<>getrs(
//    cusolverDnHandle_t handle,
//    cublasOperation_t trans,
//    int n, int nrhs,
//    const <> *A, int lda,
//    const int *devIpiv,
//    <> *B, int ldb,
//    int *devInfo );

template<typename T>
struct getrs_func_def_t
{
    typedef cusolverStatus_t (*getrs_func_def) (
                              cusolverDnHandle_t,
                              cublasOperation_t,
                              int, int,
                              const T *, int,
                              const int *,
                              T *, int,
                              int *);
};

#define SOLVE_FUNC_DEF( FUNC )                                                  \
template<typename T>                                                            \
typename FUNC##_func_def_t<T>::FUNC##_func_def                                  \
FUNC##_func();

#define SOLVE_FUNC( FUNC, TYPE, PREFIX )                                                           \
template<> typename FUNC##_func_def_t<TYPE>::FUNC##_func_def FUNC##_func<TYPE>()                \
{ return &cusolverDn##PREFIX##FUNC; }                                                           \

SOLVE_FUNC_DEF( getrs )
SOLVE_FUNC(getrs , float  , S)
SOLVE_FUNC(getrs , double , D)
SOLVE_FUNC(getrs , cfloat , C)
SOLVE_FUNC(getrs , cdouble, Z)

//cusolverStatus_t cusolverDn<>geqrf_bufferSize(
//        cusolverDnHandle_t handle,
//        int m, int n,
//        <> *A,
//        int lda,
//        int *Lwork );
//
//cusolverStatus_t cusolverDn<>geqrf(
//        cusolverDnHandle_t handle,
//        int m, int n,
//        <> *A, int lda,
//        <> *TAU,
//        <> *Workspace,
//        int Lwork, int *devInfo );
//
//cusolverStatus_t cusolverDn<>mqr(
//        cusolverDnHandle_t handle,
//        cublasSideMode_t side, cublasOperation_t trans,
//        int m, int n, int k,
//        const double *A, int lda,
//        const double *tau,
//        double *C, int ldc,
//        double *work,
//        int lwork, int *devInfo);

template<typename T>
struct geqrf_func_def_t
{
    typedef cusolverStatus_t (*geqrf_func_def) (
                              cusolverDnHandle_t, int, int,
                              T *, int,
                              T *,
                              T *,
                              int, int *);
};

template<typename T>
struct geqrf_buf_func_def_t
{
    typedef cusolverStatus_t (*geqrf_buf_func_def) (
                              cusolverDnHandle_t, int, int,
                              T *, int, int *);
};

template<typename T>
struct mqr_func_def_t
{
    typedef cusolverStatus_t (*mqr_func_def) (
                              cusolverDnHandle_t,
                              cublasSideMode_t, cublasOperation_t,
                              int, int, int,
                              const T *, int,
                              const T *,
                              T *, int,
                              T *, int,
                              int *);
};

#define QR_FUNC_DEF( FUNC )                                                     \
template<typename T>                                                            \
static typename FUNC##_func_def_t<T>::FUNC##_func_def                                  \
FUNC##_func();                                                                  \
                                                                                \
template<typename T>                                                            \
static typename FUNC##_buf_func_def_t<T>::FUNC##_buf_func_def                          \
FUNC##_buf_func();                                                              \

#define QR_FUNC( FUNC, TYPE, PREFIX )                                                           \
template<> typename FUNC##_func_def_t<TYPE>::FUNC##_func_def FUNC##_func<TYPE>()                \
{ return &cusolverDn##PREFIX##FUNC; }                                                           \
                                                                                                \
template<> typename FUNC##_buf_func_def_t<TYPE>::FUNC##_buf_func_def FUNC##_buf_func<TYPE>()    \
{ return & cusolverDn##PREFIX##FUNC##_bufferSize; }

QR_FUNC_DEF( geqrf )
QR_FUNC(geqrf , float  , S)
QR_FUNC(geqrf , double , D)
QR_FUNC(geqrf , cfloat , C)
QR_FUNC(geqrf , cdouble, Z)

#define MQR_FUNC_DEF( FUNC )                                                    \
template<typename T>                                                            \
static typename FUNC##_func_def_t<T>::FUNC##_func_def                                  \
FUNC##_func();

#define MQR_FUNC( FUNC, TYPE, PREFIX )                                                          \
template<> typename FUNC##_func_def_t<TYPE>::FUNC##_func_def FUNC##_func<TYPE>()                \
{ return &cusolverDn##PREFIX; }                                                                 \

MQR_FUNC_DEF( mqr )
MQR_FUNC(mqr , float  , Sormqr)
MQR_FUNC(mqr , double , Dormqr)
MQR_FUNC(mqr , cfloat , Cunmqr)
MQR_FUNC(mqr , cdouble, Zunmqr)

template<typename T>
Array<T> solve_square(const Array<T> &a, const Array<T> &b, const af_solve_t options)
{
    int M = a.dims()[0];
    int N = a.dims()[1];
    int K = b.dims()[1];

    Array<T> A = copyArray<T>(a);
    Array<T> B = copyArray<T>(b);
    Array<int> pivot = lu_inplace(A, false);

    int *info = memAlloc<int>(1);

    CUSOLVER_CHECK(getrs_func<T>()(getDnHandle(),
                                   CUBLAS_OP_N,
                                   N, K,
                                   A.get(), A.strides()[1],
                                   pivot.get(),
                                   B.get(), B.strides()[1],
                                   info));
    return B;
}

template<typename T>
Array<T> solve_rect(const Array<T> &a, const Array<T> &b, const af_solve_t options)
{
    int M = a.dims()[0];
    int N = a.dims()[1];
    int K = b.dims()[1];

    Array<T> B = createEmptyArray<T>(dim4());

    if (M < N) {

        Array<T> A = transpose<T>(a, true);
        B = padArray<T, T>(b, dim4(N, K), scalar<T>(0));

        /*

          The following code does the in place equivalent of the following

          Array<T> Q = createEmptyArray<T>(dim4());
          Array<T> R = createEmptyArray<T>(dim4());
          Array<T> Tau = createEmptyArray<T>(dim4());
          qr(Q, R, Tau, A);
          R.resetDims(dim4(M, M));
          B.resetDims(dim4(M, K));
          trsm<T>(R, B, AF_CONJUGATE_TRANSPOSE, true, true, false);
          B.resetDims(dim4(N, K));
          B = matmul<T>(Q, B, AF_NO_TRANSPOSE, AF_NO_TRANSPOSE);
        */

        int lwork = 0;

        CUSOLVER_CHECK(geqrf_buf_func<T>()(getDnHandle(),
                                           A.dims()[0], A.dims()[1],
                                           A.get(), A.strides()[1],
                                           &lwork));

        T *workspace = memAlloc<T>(lwork);

        Array<T> t = createEmptyArray<T>(af::dim4(min(M, N), 1, 1, 1));
        int *info = memAlloc<int>(1);
        CUSOLVER_CHECK(geqrf_func<T>()(getDnHandle(),
                                       A.dims()[0], A.dims()[1],
                                       A.get(), A.strides()[1],
                                       t.get(),
                                       workspace, lwork,
                                       info));

        A.resetDims(dim4(M, M));
        B.resetDims(dim4(M, K));
        trsm<T>(A, B, AF_CONJUGATE_TRANSPOSE, true, true, false);

        B.resetDims(dim4(N, K));

        CUSOLVER_CHECK(mqr_func<T>()(getDnHandle(),
                                     CUBLAS_SIDE_LEFT, CUBLAS_OP_N,
                                     B.dims()[0],
                                     B.dims()[1],
                                     A.dims()[0],
                                     A.get(), A.strides()[1],
                                     t.get(),
                                     B.get(), B.strides()[1],
                                     workspace, lwork,
                                     info));
    } else if (M > N) {
        B = createValueArray<T>(dim4(N, K), scalar<T>(0));
    }
    return B;
}

template<typename T>
Array<T> solve(const Array<T> &a, const Array<T> &b, const af_solve_t options)
{
    if(a.dims()[0] == a.dims()[1]) {
        return solve_square<T>(a, b, options);
    } else {
        return solve_rect<T>(a, b, options);
    }
}

#define INSTANTIATE_SOLVE(T)                                                                   \
    template Array<T> solve<T> (const Array<T> &a, const Array<T> &b, const af_solve_t options);

INSTANTIATE_SOLVE(float)
INSTANTIATE_SOLVE(cfloat)
INSTANTIATE_SOLVE(double)
INSTANTIATE_SOLVE(cdouble)

}

#else
namespace cuda
{

template<typename T>
Array<T> solve(const Array<T> &a, const Array<T> &b, const af_solve_t options)
{
    AF_ERROR("CUDA cusolver not available. Linear Algebra is disabled",
              AF_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE_SOLVE(T)                                                                   \
    template Array<T> solve<T> (const Array<T> &a, const Array<T> &b, const af_solve_t options);

INSTANTIATE_SOLVE(float)
INSTANTIATE_SOLVE(cfloat)
INSTANTIATE_SOLVE(double)
INSTANTIATE_SOLVE(cdouble)

}

#endif
