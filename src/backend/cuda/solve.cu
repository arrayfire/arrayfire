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

#if defined(WITH_CUDA_LINEAR_ALGEBRA)

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

#define SOLVE_FUNC( FUNC, TYPE, PREFIX )                                                    \
template<> typename FUNC##_func_def_t<TYPE>::FUNC##_func_def FUNC##_func<TYPE>()            \
{ return (FUNC##_func_def_t<TYPE>::FUNC##_func_def)&cusolverDn##PREFIX##FUNC; }             \

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
struct geqrf_solve_func_def_t
{
    typedef cusolverStatus_t (*geqrf_solve_func_def) (
                              cusolverDnHandle_t, int, int,
                              T *, int,
                              T *,
                              T *,
                              int, int *);
};

template<typename T>
struct geqrf_solve_buf_func_def_t
{
    typedef cusolverStatus_t (*geqrf_solve_buf_func_def) (
                              cusolverDnHandle_t, int, int,
                              T *, int, int *);
};

template<typename T>
struct mqr_solve_func_def_t
{
    typedef cusolverStatus_t (*mqr_solve_func_def) (
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
static typename FUNC##_solve_func_def_t<T>::FUNC##_solve_func_def               \
FUNC##_solve_func();                                                            \
                                                                                \
template<typename T>                                                            \
static typename FUNC##_solve_buf_func_def_t<T>::FUNC##_solve_buf_func_def       \
FUNC##_solve_buf_func();                                                        \

#define QR_FUNC( FUNC, TYPE, PREFIX )                                                                               \
template<> typename FUNC##_solve_func_def_t<TYPE>::FUNC##_solve_func_def FUNC##_solve_func<TYPE>()                  \
{ return (FUNC##_solve_func_def_t<TYPE>::FUNC##_solve_func_def)&cusolverDn##PREFIX##FUNC; }                         \
                                                                                                                    \
template<> typename FUNC##_solve_buf_func_def_t<TYPE>::FUNC##_solve_buf_func_def FUNC##_solve_buf_func<TYPE>()      \
{ return (FUNC##_solve_buf_func_def_t<TYPE>::FUNC##_solve_buf_func_def)& cusolverDn##PREFIX##FUNC##_bufferSize; }

QR_FUNC_DEF( geqrf )
QR_FUNC(geqrf , float  , S)
QR_FUNC(geqrf , double , D)
QR_FUNC(geqrf , cfloat , C)
QR_FUNC(geqrf , cdouble, Z)

#define MQR_FUNC_DEF( FUNC )                                                            \
template<typename T>                                                                    \
static typename FUNC##_solve_func_def_t<T>::FUNC##_solve_func_def                       \
FUNC##_solve_func();

#define MQR_FUNC( FUNC, TYPE, PREFIX )                                                  \
template<> typename FUNC##_solve_func_def_t<TYPE>::FUNC##_solve_func_def                \
FUNC##_solve_func<TYPE>()                                                               \
{ return (FUNC##_solve_func_def_t<TYPE>::FUNC##_solve_func_def)&cusolverDn##PREFIX; }   \

MQR_FUNC_DEF( mqr )
MQR_FUNC(mqr , float  , Sormqr)
MQR_FUNC(mqr , double , Dormqr)
MQR_FUNC(mqr , cfloat , Cunmqr)
MQR_FUNC(mqr , cdouble, Zunmqr)

template<typename T>
Array<T> solveLU(const Array<T> &A, const Array<int> &pivot,
                 const Array<T> &b, const af_mat_prop options)
{
    int N = A.dims()[0];
    int NRHS = b.dims()[1];

    Array< T > B = copyArray<T>(b);

    int *info = memAlloc<int>(1);

    CUSOLVER_CHECK(getrs_func<T>()(getDnHandle(),
                                   CUBLAS_OP_N,
                                   N, NRHS,
                                   A.get(), A.strides()[1],
                                   pivot.get(),
                                   B.get(), B.strides()[1],
                                   info));

    memFree(info);
    return B;
}

template<typename T>
Array<T> generalSolve(const Array<T> &a, const Array<T> &b)
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
    memFree(info);
    return B;
}

template<typename T>
cublasOperation_t trans() { return CUBLAS_OP_T; }
template<> cublasOperation_t trans<cfloat>() { return CUBLAS_OP_C; }
template<> cublasOperation_t trans<cdouble>() { return CUBLAS_OP_C; }


template<typename T>
Array<T> leastSquares(const Array<T> &a, const Array<T> &b)
{
    int M = a.dims()[0];
    int N = a.dims()[1];
    int K = b.dims()[1];

    Array<T> B = createEmptyArray<T>(dim4());

    if (M < N) {

        // Least squres for this case is solved using the following
        // solve(A, B) == matmul(Q, Xpad);
        // Where:
        // Xpad == pad(Xt, N - M, 1);
        // Xt   == tri_solve(R1, B);
        // R1   == R(seq(M), seq(M));
        // transpose(A) == matmul(Q, R);

        // QR is performed on the transpose of A
        Array<T> A = transpose<T>(a, true);
        B = padArray<T, T>(b, dim4(N, K), scalar<T>(0));

        int lwork = 0;

        // Get workspace needed for QR
        CUSOLVER_CHECK(geqrf_solve_buf_func<T>()(getDnHandle(),
                                                 A.dims()[0], A.dims()[1],
                                                 A.get(), A.strides()[1],
                                                 &lwork));

        T *workspace = memAlloc<T>(lwork);
        Array<T> t = createEmptyArray<T>(af::dim4(min(M, N), 1, 1, 1));
        int *info = memAlloc<int>(1);

        // In place Perform in place QR
        CUSOLVER_CHECK(geqrf_solve_func<T>()(getDnHandle(),
                                             A.dims()[0], A.dims()[1],
                                             A.get(), A.strides()[1],
                                             t.get(),
                                             workspace, lwork,
                                             info));

        // R1 = R(seq(M), seq(M));
        A.resetDims(dim4(M, M));

        // Bt = tri_solve(R1, B);
        B.resetDims(dim4(M, K));
        trsm<T>(A, B, AF_MAT_CTRANS, true, true, false);

        // Bpad = pad(Bt, ..)
        B.resetDims(dim4(N, K));

        // matmul(Q, Bpad)
        CUSOLVER_CHECK(mqr_solve_func<T>()(getDnHandle(),
                                           CUBLAS_SIDE_LEFT, CUBLAS_OP_N,
                                           B.dims()[0],
                                           B.dims()[1],
                                           A.dims()[0],
                                           A.get(), A.strides()[1],
                                           t.get(),
                                           B.get(), B.strides()[1],
                                           workspace, lwork,
                                           info));

        memFree(workspace);
        memFree(info);

    } else if (M > N) {

        // Least squres for this case is solved using the following
        // solve(A, B) == tri_solve(R1, Bt);
        // Where:
        // R1 == R(seq(N), seq(N));
        // Bt == matmul(transpose(Q1), B);
        // Q1 == Q(span, seq(N));
        // A  == matmul(Q, R);

        Array<T> A = copyArray<T>(a);
        B = copyArray(b);

        int lwork = 0;

        // Get workspace needed for QR
        CUSOLVER_CHECK(geqrf_solve_buf_func<T>()(getDnHandle(),
                                                 A.dims()[0], A.dims()[1],
                                                 A.get(), A.strides()[1],
                                                 &lwork));

        T *workspace = memAlloc<T>(lwork);
        Array<T> t = createEmptyArray<T>(af::dim4(min(M, N), 1, 1, 1));
        int *info = memAlloc<int>(1);

        // In place Perform in place QR
        CUSOLVER_CHECK(geqrf_solve_func<T>()(getDnHandle(),
                                             A.dims()[0], A.dims()[1],
                                             A.get(), A.strides()[1],
                                             t.get(),
                                             workspace, lwork,
                                             info));

        // matmul(Q1, B)
        CUSOLVER_CHECK(mqr_solve_func<T>()(getDnHandle(),
                                           CUBLAS_SIDE_LEFT,
                                           trans<T>(),
                                           M, K, N,
                                           A.get(), A.strides()[1],
                                           t.get(),
                                           B.get(), B.strides()[1],
                                           workspace, lwork,
                                           info));

        // tri_solve(R1, Bt)
        A.resetDims(dim4(N, N));
        B.resetDims(dim4(N, K));
        trsm(A, B, AF_MAT_NONE, true, true, false);

        memFree(workspace);
        memFree(info);
    }
    return B;
}

template<typename T>
Array<T> triangleSolve(const Array<T> &A, const Array<T> &b, const af_mat_prop options)
{
    Array<T> B = copyArray<T>(b);
    trsm(A, B,
         AF_MAT_NONE, // transpose flag
         options & AF_MAT_UPPER ? true : false,
         true, // is_left
         options & AF_MAT_DIAG_UNIT ? true : false);
    return B;
}

template<typename T>
Array<T> solve(const Array<T> &a, const Array<T> &b, const af_mat_prop options)
{
    if (options & AF_MAT_UPPER ||
        options & AF_MAT_LOWER) {
        return triangleSolve<T>(a, b, options);
    }

    if(a.dims()[0] == a.dims()[1]) {
        return generalSolve<T>(a, b);
    } else {
        return leastSquares<T>(a, b);
    }
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
namespace cuda
{

template<typename T>
Array<T> solveLU(const Array<T> &A, const Array<int> &pivot,
                 const Array<T> &b, const af_mat_prop options)
{
    AF_ERROR("Linear Algebra is diabled on CUDA",
             AF_ERR_NOT_CONFIGURED);
}

template<typename T>
Array<T> solve(const Array<T> &a, const Array<T> &b, const af_mat_prop options)
{
    AF_ERROR("Linear Algebra is diabled on CUDA",
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
