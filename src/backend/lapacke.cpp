/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#if defined(__APPLE__) && !defined(AF_CUDA)
#include <Accelerate/Accelerate.h>
#include "lapacke.hpp"
#include <cstdint>
#include <algorithm>

#if INTPTR_MAX == INT16MAX
    #define BS 16
#elif INTPTR_MAX == INT32MAX
    #define BS 32
#elif INTPTR_MAX == INT64MAX
    #define BS 64
#else
    #define BS 32
#endif

#define LAPACK_FUNC(X, T, TO)                                                       \
int LAPACKE_##X##geqrf(int layout, int M, int N, T *A, int lda, T *tau)             \
{                                                                                   \
    int lwork = N * BS;                                                             \
    T *work = new T[lwork];                                                         \
    int info = 0;                                                                   \
    X##geqrf_(&M, &N, (TO)A, &lda, (TO)tau, (TO)work, &lwork, &info);               \
    delete [] work;                                                                 \
    return info;                                                                    \
}                                                                                   \
int LAPACKE_##X##geqrf_work(int layout, int M, int N, T *A, int lda,                \
                            T *tau, T *work, int lwork)                             \
{                                                                                   \
    int info = 0;                                                                   \
    X##geqrf_(&M, &N, (TO)A, &lda, (TO)tau, (TO)work, &lwork, &info);               \
    return info;                                                                    \
}                                                                                   \
int LAPACKE_##X##getrf(int layout, int M, int N, T *A, int lda, int *pivot)         \
{                                                                                   \
    int info = 0;                                                                   \
    X##getrf_(&M, &N, (TO)A, &lda, pivot, &info);                                   \
    return info;                                                                    \
}                                                                                   \
int LAPACKE_##X##getrs(int layout, char trans, int M, int N, const T *A,            \
                       int lda, const int *pivot, T *B, int ldb)                    \
{                                                                                   \
    int info = 0;                                                                   \
    X##getrs_(&trans, &M, &N, (TO)A, &lda, (int *)pivot, (TO)B, &ldb, &info);       \
    return info;                                                                    \
}                                                                                   \
int LAPACKE_##X##potrf(int layout, char uplo, int N, T *A, int lda)                 \
{                                                                                   \
    int info = 0;                                                                   \
    X##potrf_(&uplo, &N, (TO)A, &lda, &info);                                       \
    return info;                                                                    \
}                                                                                   \
int LAPACKE_##X##gesv(int layout, int N, int nrhs, T *A, int lda,                   \
                      int *pivot, T *B, int ldb)                                    \
{                                                                                   \
    int info = 0;                                                                   \
    X##gesv_(&N, &nrhs, (TO)A, &lda, pivot, (TO)B, &ldb, &info);                    \
    return info;                                                                    \
}                                                                                   \
int LAPACKE_##X##gels(int layout, char trans, int M, int N, int nrhs,               \
                      T *A, int lda, T *B, int ldb)                                 \
{                                                                                   \
    int lwork = std::min(M, N) + std::max(M, std::max(N, nrhs)) * BS;               \
    T *work = new T[lwork];                                                         \
    int info = 0;                                                                   \
    X##gels_(&trans, &M, &N, &nrhs, (TO)A, &lda,                                    \
                       (TO)B, &ldb, (TO)work, &lwork, &info);                       \
    delete [] work;                                                                 \
    return info;                                                                    \
}                                                                                   \
int LAPACKE_##X##getri(int layout, int N, T *A, int lda, const int *pivot)          \
{                                                                                   \
    int lwork = N * BS;                                                             \
    T *work = new T[lwork];                                                         \
    int info = 0;                                                                   \
    X##getri_(&N, (TO)A, &lda, const_cast<int *>(pivot),                            \
                        (TO)work, &lwork, &info);                                   \
    delete [] work;                                                                 \
    return info;                                                                    \
}                                                                                   \
int LAPACKE_##X##trtri(int layout, char uplo, char diag, int N, T *A, int lda)      \
{                                                                                   \
    int info = 0;                                                                   \
    X##trtri_(&uplo, &diag, &N, (TO)A, &lda, &info);                                \
    return info;                                                                    \
}                                                                                   \
int LAPACKE_##X##trtrs(int layout, char uplo, char trans, char diag,                \
                       int N, int NRHS, const T *A, int lda, T *B, int ldb)         \
{                                                                                   \
    int info = 0;                                                                   \
    X##trtrs_(&uplo, &trans, &diag, &N, &NRHS, (TO)A, &lda, (TO)B, &ldb, &info);    \
    return info;                                                                    \
}                                                                                   \
int LAPACKE_##X##larft(int layout, char direct, char storev, int N, int K,          \
                       const T *v, int ldv, const T *tau, T *t, int ldt)            \
{                                                                                   \
    X##larft_(&direct, &storev, &N, &K, (TO)v, &ldv,                                \
                        (TO)const_cast<T*>(tau), (TO)t, &ldt);                      \
    return 0;                                                                       \
}                                                                                   \
int LAPACKE_##X##laswp(int layout, int N, T *A, int lda,                            \
                       int k1, int k2, const int *pivot, int incx)                  \
{                                                                                   \
    X##laswp_(&N, (TO)A, &lda, &k1, &k2, const_cast<int*>(pivot), &incx);           \
    return 0;                                                                       \
}                                                                                   \

LAPACK_FUNC(s, float, float*)
LAPACK_FUNC(d, double, double*)
LAPACK_FUNC(c, cfloat, __CLPK_complex*)
LAPACK_FUNC(z, cdouble, __CLPK_doublecomplex*)

#define LAPACK_GQR(P, X, T, TO)                                                     \
int LAPACKE_##X##P(int layout, int M, int N, int K, T *A, int lda, const T *tau)    \
{                                                                                   \
    int lwork = N * 32;                                                             \
    T *work = new T[lwork];                                                         \
    int info = 0;                                                                   \
    X##P##_(&M, &N, &K, (TO)A, &lda, (TO)tau, (TO)work, &lwork, &info);             \
    delete [] work;                                                                 \
    return info;                                                                    \
}                                                                                   \

LAPACK_GQR(orgqr, s, float, float*)
LAPACK_GQR(orgqr, d, double, double*)
LAPACK_GQR(ungqr, c, cfloat, __CLPK_complex*)
LAPACK_GQR(ungqr, z, cdouble, __CLPK_doublecomplex*)

#define LAPACK_GQR_WORK(P, X, T, TO)                                                \
int LAPACKE_##X##P##_work(int layout, int M, int N, int K, T *A, int lda,           \
                          const T *tau, T *work, int lwork)                         \
{                                                                                   \
    int info = 0;                                                                   \
    X##P##_(&M, &N, &K, (TO)A, &lda, (TO)tau, (TO)work, &lwork, &info);             \
    return info;                                                                    \
}                                                                                   \

LAPACK_GQR_WORK(orgqr, s, float, float*)
LAPACK_GQR_WORK(orgqr, d, double, double*)
LAPACK_GQR_WORK(ungqr, c, cfloat, __CLPK_complex*)
LAPACK_GQR_WORK(ungqr, z, cdouble, __CLPK_doublecomplex*)

#define LAPACK_MQR_WORK(P, X, T, TO)                                                \
int LAPACKE_##X##P##_work(int layout, char side, char trans, int M, int N, int K,   \
                          const T *A, int lda, const T *tau, T *c, int ldc,         \
                          T *work, int lwork)                                       \
{                                                                                   \
    int info = 0;                                                                   \
    X##P##_(&side, &trans, &M, &N, &K, (TO)A, &lda, (TO)tau, (TO)c, &ldc,           \
                      (TO)work, &lwork, &info);                                     \
    return info;                                                                    \
}                                                                                   \

LAPACK_MQR_WORK(ormqr, s, float, float*)
LAPACK_MQR_WORK(ormqr, d, double, double*)
LAPACK_MQR_WORK(unmqr, c, cfloat, __CLPK_complex*)
LAPACK_MQR_WORK(unmqr, z, cdouble, __CLPK_doublecomplex*)

#endif
