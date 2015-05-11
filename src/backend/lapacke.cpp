/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#ifdef USE_CLAPACK
#include <cstdint>
#include <lapack.h>

#if INTPTR_MAX == INT16MAX
    #define BLOCK_SIZE 16
#else INTPTR_MAX == INT32MAX
    #define BLOCK_SIZE 32
#else INTPTR_MAX == INT64MAX
    #define BLOCK_SIZE 64
#endif
#define BS BLOCK_SIZE

#define LAPACK(X, T)                                                                \
int LAPACKE_##X##geqrf(int layout, int M, int N, T *A, int lda, T *tau)             \
{                                                                                   \
    int lwork = N * BS;                                                             \
    T *work = new T[lwork];                                                         \
    int info = 0;                                                                   \
    int ret = X##geqrf_(&M, &N, A, &lda, tau, work, &lwork, &info);                 \
    delete [] work;                                                                 \
    return info;                                                                    \
}                                                                                   \
int LAPACKE_##X##geqrf_work(int layout, int M, int N, T *A, int lda,                \
                            T *tau, T *work. int lwork)                             \
{                                                                                   \
    int info = 0;                                                                   \
    int ret = X##geqrf_(&M, &N, A, &lda, tau, work, &lwork, &info);                 \
    return info;                                                                    \
}                                                                                   \
int LAPACKE_##X##getrf(int layout, int M, int N, T *A, int lda, int *pivot)         \
{                                                                                   \
    int info = 0;                                                                   \
    int ret = X##getrf_(&M, &N, A, &lda, pivot, &info);                             \
    return info;                                                                    \
}                                                                                   \
int LAPACKE_##X##potrf(int layout, char uplo, int N, T *A, int lda)                 \
{                                                                                   \
    int info = 0;                                                                   \
    int ret = X##potrf_(&uplo, &N, A, &lda, &info);                                 \
    return info;                                                                    \
}                                                                                   \
int LAPACKE_##X##gesv(int layout, int N, int nrhs, T *A, int lda,                   \
                      int *pivot, T *B, int ldb)                                    \
{                                                                                   \
    int info = 0;                                                                   \
    int ret = X##gesv_(&N, &nrhs, A, &lda, &pivot, B, &ldb, &info);                 \
    return info;                                                                    \
}                                                                                   \
int LAPACKE_##X##gels(int layout, char trans, int M, int N, int nrhs,               \
                      T *A, int lda, T *B, int ldb)                                 \
{                                                                                   \
    int lwork = std::min(M, N) + std::max(M, std::max(N, nrhs)) * BS;               \
    T *work = new T[lwork];                                                         \
    int info = 0;                                                                   \
    int ret = X##gels_(&trans, &M, &N, &nrhs, A, &lda, B, &ldb, work, &lwork, &info);\
    delete [] work;                                                                 \
    return info;                                                                    \
}                                                                                   \
int LAPACKE_##X##getri(int layout, int N, T *A, int lda, const int *pivot)          \
{                                                                                   \
    int lwork = N * BS;                                                             \
    T *work = new T[lwork];                                                         \
    int info = 0;                                                                   \
    int ret = X##getri_(&N, A, &lda, pivot, work, &lwork, &info);                   \
    delete [] work;                                                                 \
    return info;                                                                    \
}                                                                                   \
int LAPACKE_##X##trtri(int layout, char uplo, char diag, int N, T *A, int lda)      \
{                                                                                   \
    int ret = X##trtri_(&uplo, &diag, &N, A, &lda, &info);                          \
    return ret;                                                                     \
}                                                                                   \
int LAPACKE_##X##larft(int layout, char direct, char storev, int N, int K,          \
                       const T *v, int ldv, const T *tau, T *t, int ldt)            \
{                                                                                   \
    int ret = X##larft_(&direct, &storev, &N, &K, v, &ldv, tau, t, &ldt, &info);    \
    return ret;                                                                     \
}                                                                                   \
int LAPACKE_##X##laswp(int layout, int N, int K, T *A, int lda,                     \
                       int k1, int k2, const int * pivot, int incx)                 \
{                                                                                   \
    int ret = X##laswp_(&N, A, &lda, &k1, &k2, pivot, incx);                        \
    return ret;                                                                     \
}                                                                                   \

LAPACK(s, float)
LAPACK(d, double)
LAPACK(c, cfloat)
LAPACK(z, cdouble)

#define LAPACK_GQR(P, X, T)                                                       \
int LAPACKE_##X##P(int layout, int M, int N, int K, T *A, int lda, const T *tau)    \
{                                                                                   \
    int lwork = N * 32 ;                                                            \
    T *work = new T[lwork];                                                         \
    int info = 0;                                                                   \
    int ret = X##P##_(&M, &N, A, &lda, tau, work, &lwork, &info);                   \
    delete [] work;                                                                 \
    return info;                                                                    \
}                                                                                   \

LAPACK_GQR(orgqr, s, float)
LAPACK_GQR(orgqr, d, double)
LAPACK_GQR(ungqr, c, cfloat)
LAPACK_GQR(ungqr, z, cdouble)

#define LAPACK_GQR_WORK(P, X, T)                                                    \
int LAPACKE_##X##P##_work(int layout, int M, int N, int K, T *A, int lda,           \
                          const T *tau, T *work, int lwork)                         \
{                                                                                   \
    int info = 0;                                                                   \
    int ret = X##P##_(&M, &N, &K, A, &lda, tau, work, &lwork, &info);               \
    return info;                                                                    \
}                                                                                   \

LAPACK_GQR_WORK(orgqr, s, float)
LAPACK_GQR_WORK(orgqr, d, double)
LAPACK_GQR_WORK(ungqr, c, cfloat)
LAPACK_GQR_WORK(ungqr, z, cdouble)

#define LAPACK_MQR_WORK(P, X, T)                                                    \
int LAPACKE_##X##P##_work(int layout, char side, char trans, int M, int N, int K,   \
                          const T *A, int lda, const T *tau, T *c, int ldc,         \
                          T *work, int lwork)                                       \
{                                                                                   \
    int info = 0;                                                                   \
    int ret = X##P##_(&side, &trans, &M, &N, &K, A, &lda, tau, c, &ldc,             \
                      work, &lwork, &info);                                         \
    return info;                                                                    \
}                                                                                   \

LAPACK_MQR_WORK(ormqr, s, float)
LAPACK_MQR_WORK(ormqr, d, double)
LAPACK_MQR_WORK(unmqr, c, cfloat)
LAPACK_MQR_WORK(unmqr, z, cdouble)

#endif
