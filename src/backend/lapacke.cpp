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
#include <traits.hpp>
#include <vector>

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


#define LAPACK_GESDD_REAL(P, X, T, Tr, TO)          \
    int LAPACKE_##X##P(int layout,                  \
                       char jobz,                   \
                       int m, int n,                \
                       T* in, int ldin,             \
                       Tr* s,                       \
                       T* u, int ldu,               \
                       T* vt, int ldvt)             \
    {                                               \
        int info = 0;                               \
        int lwork = -1;                             \
        T work_param = 0;                           \
        X##P##_(&jobz, &m, &n, (TO)in, &ldin,       \
                s, (TO)u, &ldu, (TO)vt, &ldvt,      \
                &work_param, &lwork,                \
                NULL, &info);                       \
        lwork = work_param;                         \
        std::vector<T> work(lwork);                 \
        std::vector<int> iwork(8 * std::min(m, n)); \
        X##P##_(&jobz, &m, &n, (TO)in, &ldin,       \
                s, (TO)u, &ldu, (TO)vt, &ldvt,      \
                (TO)&work[0], &lwork,               \
                &iwork[0], &info);                  \
        return info;                                \
    }

#define LAPACK_GESDD_CPLX(P, X, T, Tr, TO)          \
    int LAPACKE_##X##P(int layout,                  \
                       char jobz,                   \
                       int m, int n,                \
                       T* in, int ldin,             \
                       Tr* s,                       \
                       T* u, int ldu,               \
                       T* vt, int ldvt)             \
    {                                               \
        int info = 0;                               \
        int max_mn = std::max(m, n);                \
        int min_mn = std::max(m, n);                \
        int lwork = 5 * max_mn;                     \
        std::vector<T> work(lwork);                 \
        std::vector<int> iwork(8 * std::min(m, n)); \
        int irwork = std::max(1,                    \
                              min_mn *              \
                              std::max(5*min_mn+7,  \
                                       2*max_mn+2*  \
                                       min_mn+1));  \
        std::vector<Tr> rwork(irwork);              \
        X##P##_(&jobz, &m, &n, (TO)in, &ldin,       \
                s, (TO)u, &ldu, (TO)vt, &ldvt,      \
                (TO)&work[0], &lwork,               \
                &rwork[0], &iwork[0], &info);       \
        return info;                                \
    }


LAPACK_GESDD_REAL(gesdd, s, float  , float , float*)
LAPACK_GESDD_REAL(gesdd, d, double , double, double*)
LAPACK_GESDD_CPLX(gesdd, c, cfloat , float ,__CLPK_complex*)
LAPACK_GESDD_CPLX(gesdd, z, cdouble, double,__CLPK_doublecomplex*)

#define LAPACK_LAMCH(X, T) T LAPACKE_##X##lamch(char cmach) { return X##lamch_(&cmach); }

LAPACK_LAMCH(s, float )
LAPACK_LAMCH(d, double)

#define LAPACK_LACPY(X, T, TO)                                  \
    int LAPACKE_##X##lacpy(int matrix_order, char uplo, int m,  \
                           int n, const T* a,                   \
                           int lda, T* b,                       \
                           int ldb )                            \
    {                                                           \
        int info = 0;                                           \
        X##lacpy_(&uplo, &m, &n, (TO)a, &lda, (TO)b, &ldb);     \
        return info;                                            \
    }

LAPACK_LACPY(s, float, float*)
LAPACK_LACPY(d, double, double*)
LAPACK_LACPY(c, cfloat,__CLPK_complex*)
LAPACK_LACPY(z, cdouble,__CLPK_doublecomplex*)

#define LAPACK_GBR_WORK(P, X, T, TO)                                \
    int LAPACKE_##X##P##_work(int matrix_order, char vect, int m,   \
                              int n, int k, T* a,                   \
                              int lda, const T* tau, T* work,       \
                              int lwork )                           \
    {                                                               \
        int info = 0;                                               \
        X##P##_(&vect, &m, &n, &k, (TO)a, &lda,                     \
                (TO)tau, (TO)work, &lwork, &info);                  \
        return info;                                                \
    }

LAPACK_GBR_WORK(orgbr, s, float, float*)
LAPACK_GBR_WORK(orgbr, d, double, double*)
LAPACK_GBR_WORK(ungbr, c, cfloat,__CLPK_complex*)
LAPACK_GBR_WORK(ungbr, z, cdouble,__CLPK_doublecomplex*)

#define LAPACK_BDSQR_WORK(X, T, Tr, TO)                                 \
    int LAPACKE_##X##bdsqr_work( int matrix_order, char uplo, int n,    \
                                 int ncvt, int nru, int ncc,            \
                                 Tr* d, Tr* e, T* vt,                   \
                                 int ldvt, T* u,                        \
                                 int ldu, T* c,                         \
                                 int ldc, Tr* work)                     \
    {                                                                   \
        int info = 0;                                                   \
        X##bdsqr_(&uplo, &n, &ncvt, &nru, &ncc, d, e,                   \
                  (TO)vt, &ldvt, (TO)u, &ldu,                           \
                  (TO)c, &ldc, work, &info);                            \
        return info;                                                    \
    }                                                                   \


LAPACK_BDSQR_WORK(s, float, float, float*)
LAPACK_BDSQR_WORK(d, double, double, double*)
LAPACK_BDSQR_WORK(c, cfloat, float,__CLPK_complex*)
LAPACK_BDSQR_WORK(z, cdouble, double,__CLPK_doublecomplex*)


#define LAPACK_GEBRD_WORK(X, T, Tr, TO)                             \
    int LAPACKE_##X##gebrd_work( int matrix_order, int m, int n,    \
                                 T* a, int lda,                     \
                                 Tr* d, Tr* e, T* tauq,             \
                                 T* taup,                           \
                                 T* work, int lwork )               \
    {                                                               \
        int info = 0;                                               \
        X##gebrd_(&m, &n, (TO)a, &lda, d, e, (TO)tauq, (TO)taup,    \
                  (TO)work, &lwork, &info);                         \
        return info;                                                \
    }

LAPACK_GEBRD_WORK(s, float, float, float*)
LAPACK_GEBRD_WORK(d, double, double, double*)
LAPACK_GEBRD_WORK(c, cfloat, float, __CLPK_complex*)
LAPACK_GEBRD_WORK(z, cdouble, double,__CLPK_doublecomplex*)

#define LAPACK_LARFG_WORK(X, T, TO)                         \
    int LAPACKE_##X##larfg_work( int n, T* alpha,           \
                                 T* x, int incx,            \
                                 T* tau )                   \
    {                                                       \
        int info = 0;                                       \
        X##larfg_(&n, (TO)alpha, (TO)x, &incx, (TO)tau);    \
        return info;                                        \
    }

LAPACK_LARFG_WORK(s, float, float*)
LAPACK_LARFG_WORK(d, double, double*)
LAPACK_LARFG_WORK(c, cfloat, __CLPK_complex*)
LAPACK_LARFG_WORK(z, cdouble, __CLPK_doublecomplex*)

#define LAPACK_LACGV_WORK(X, T, TO)             \
    int LAPACKE_##X##lacgv_work( int n, T* x,   \
                                 int incx)      \
    {                                           \
        X##lacgv_(&n, (TO)x, &incx);            \
        return 0;                               \
    }

LAPACK_LACGV_WORK(c, cfloat, __CLPK_complex*)
LAPACK_LACGV_WORK(z, cdouble, __CLPK_doublecomplex*)


#endif
