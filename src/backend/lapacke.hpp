/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#if defined(__APPLE__)
#include <af/defines.h>
#include <types.hpp>
#include <backend.hpp>

using detail::cfloat;
using detail::cdouble;

#define LAPACK_FUNC(X, T)                                                           \
int LAPACKE_##X##geqrf(int layout, int M, int N, T *A, int lda, T *tau);            \
int LAPACKE_##X##geqrf_work(int layout, int M, int N, T *A, int lda,                \
                            T *tau, T *work, int lwork);                            \
int LAPACKE_##X##getrf(int layout, int M, int N, T *A, int lda, int *pivot);        \
int LAPACKE_##X##potrf(int layout, char uplo, int N, T *A, int lda);                \
int LAPACKE_##X##gesv(int layout, int N, int nrhs, T *A, int lda,                   \
                      int *pivot, T *B, int ldb);                                   \
int LAPACKE_##X##gels(int layout, char trans, int M, int N, int nrhs,               \
                      T *A, int lda, T *B, int ldb);                                \
int LAPACKE_##X##getri(int layout, int N, T *A, int lda, const int *pivot);         \
int LAPACKE_##X##trtri(int layout, char uplo, char diag, int N, T *A, int lda);     \
int LAPACKE_##X##larft(int layout, char direct, char storev, int N, int K,          \
                       const T *v, int ldv, const T *tau, T *t, int ldt);           \
int LAPACKE_##X##laswp(int layout, int N, T *A, int lda,                            \
                       int k1, int k2, const int * pivot, int incx);                \
int LAPACKE_##X##getrs(int layout, char trans, int M, int N, const T *A,            \
                       int lda, const int *pivot, T *B, int ldb);                   \
int LAPACKE_##X##trtrs(int layout, char uplo, char trans, char diag,                \
                       int N, int NRHS, const T *A, int lda, T *B, int ldb);        \

LAPACK_FUNC(s, float)
LAPACK_FUNC(d, double)
LAPACK_FUNC(c, cfloat)
LAPACK_FUNC(z, cdouble)

#define LAPACK_GQR(P, X, T)                                                         \
int LAPACKE_##X##P(int layout, int M, int N, int K, T *A, int lda, const T *tau);   \

LAPACK_GQR(orgqr, s, float)
LAPACK_GQR(orgqr, d, double)
LAPACK_GQR(ungqr, c, cfloat)
LAPACK_GQR(ungqr, z, cdouble)

#define LAPACK_GQR_WORK(P, X, T)                                                    \
int LAPACKE_##X##P##_work(int layout, int M, int N, int K, T *A, int lda,           \
                          const T *tau, T *work, int lwork);                        \

LAPACK_GQR_WORK(orgqr, s, float)
LAPACK_GQR_WORK(orgqr, d, double)
LAPACK_GQR_WORK(ungqr, c, cfloat)
LAPACK_GQR_WORK(ungqr, z, cdouble)

#define LAPACK_MQR_WORK(P, X, T)                                                    \
int LAPACKE_##X##P##_work(int layout, char side, char trans, int M, int N, int K,   \
                          const T *A, int lda, const T *tau, T *c, int ldc,         \
                          T *work, int lwork);                                      \

LAPACK_MQR_WORK(ormqr, s, float)
LAPACK_MQR_WORK(ormqr, d, double)
LAPACK_MQR_WORK(unmqr, c, cfloat)
LAPACK_MQR_WORK(unmqr, z, cdouble)

#define LAPACK_GESDD(P, X, T, Tr)               \
    int LAPACKE_##X##P(int layout,              \
                       char jobz,               \
                       int m, int n,            \
                       T* in, int ldin,         \
                       Tr* s,                   \
                       T* u, int ldu,           \
                       T* vt, int ldvt);        \

LAPACK_GESDD(gesdd, s, float  , float )
LAPACK_GESDD(gesdd, d, double , double)
LAPACK_GESDD(gesdd, c, cfloat , float )
LAPACK_GESDD(gesdd, z, cdouble, double)

#define LAPACK_LAMCH(X, T) T LAPACKE_##X##lamch(char cmach);

LAPACK_LAMCH(s, float )
LAPACK_LAMCH(d, double)

#define LAPACK_LACPY(X, T)                                      \
    int LAPACKE_##X##lacpy(int matrix_order, char uplo, int m,  \
                           int n, const T* a,                   \
                           int lda, T* b,                       \
                           int ldb );                           \


LAPACK_LACPY(s, float)
LAPACK_LACPY(d, double)
LAPACK_LACPY(c, cfloat)
LAPACK_LACPY(z, cdouble)

#define LAPACK_GBR_WORK(P, X, T)                                    \
    int LAPACKE_##X##P##_work(int matrix_order, char vect, int m,   \
                              int n, int k, T* a,                   \
                              int lda, const T* tau, T* work,       \
                              int lwork );                          \

LAPACK_GBR_WORK(orgbr, s, float)
LAPACK_GBR_WORK(orgbr, d, double)
LAPACK_GBR_WORK(ungbr, c, cfloat)
LAPACK_GBR_WORK(ungbr, z, cdouble)

#define LAPACK_BDSQR_WORK(X, T, Tr)                                     \
    int LAPACKE_##X##bdsqr_work( int matrix_order, char uplo, int n,    \
                                 int ncvt, int nru, int ncc,            \
                                 Tr* d, Tr* e, T* vt,                   \
                                 int ldvt, T* u,                        \
                                 int ldu, T* c,                         \
                                 int ldc, Tr* work );                   \

LAPACK_BDSQR_WORK(s, float, float)
LAPACK_BDSQR_WORK(d, double, double)
LAPACK_BDSQR_WORK(c, cfloat, float)
LAPACK_BDSQR_WORK(z, cdouble, double)

#define LAPACK_GEBRD_WORK(X, T, Tr)                                 \
    int LAPACKE_##X##gebrd_work( int matrix_order, int m, int n,    \
                                 T* a, int lda,                     \
                                 Tr* d, Tr* e, T* tauq,             \
                                 T* taup,                           \
                                 T* work, int lwork );              \

LAPACK_GEBRD_WORK(s, float, float)
LAPACK_GEBRD_WORK(d, double, double)
LAPACK_GEBRD_WORK(c, cfloat, float)
LAPACK_GEBRD_WORK(z, cdouble, double)

#define LAPACK_LARFG_WORK(X, T)                     \
    int LAPACKE_##X##larfg_work( int n, T* alpha,   \
                                 T* x, int incx,    \
                                 T* tau );          \

LAPACK_LARFG_WORK(s, float)
LAPACK_LARFG_WORK(d, double)
LAPACK_LARFG_WORK(c, cfloat)
LAPACK_LARFG_WORK(z, cdouble)

#define LAPACK_LACGV_WORK(X, T)                 \
    int LAPACKE_##X##lacgv_work( int n, T* x,   \
                                 int incx );    \

LAPACK_LACGV_WORK(c, cfloat)
LAPACK_LACGV_WORK(z, cdouble)

#undef LAPACK_FUNC
#undef LAPACK_GQR
#undef LAPACK_GQR_WORK
#undef LAPACK_MQR_WORK
#undef LAPACK_GESDD
#undef LAPACK_LAMCH
#undef LAPACK_LACPY
#undef LAPACK_GBR_WORK
#undef LAPACK_BDSQR_WORK
#undef LAPACK_GEBRD_WORK
#undef LAPACK_LARFG_WORK
#undef LAPACK_LACGV_WORK
#endif
