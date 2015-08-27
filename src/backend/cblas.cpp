/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#include <blas.hpp>

#ifdef USE_F77_BLAS
#define ADD_
#include <cblas_f77.h>

static char transChar(CBLAS_TRANSPOSE Trans)
{
    switch(Trans) {
        case CblasNoTrans:      return 'N';
        case CblasTrans:        return 'T';
        case CblasConjTrans:    return 'C';
        default:                return '\0';
    }
}

#define GEMM_F77(X, TS, TV, TY)                                 \
    void cblas_##X##gemm(                                       \
        const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA,  \
        const CBLAS_TRANSPOSE TransB, const int M, const int N, \
        const int K, const TS alpha, const TV *A,               \
        const int lda, const TV *B, const int ldb,              \
        const TS beta, TV *C, const int ldc)                    \
    {                                                           \
        char aT = transChar(TransA);                            \
        char bT = transChar(TransB);                            \
        X##gemm_(&aT, &bT, &M, &N, &K,                          \
                 (const TY *)ADDR(alpha), (const TY *)A, &lda,  \
                 (const TY *)B, &ldb,                           \
                 (const TY *)ADDR(beta), (TY *)C, &ldc);        \
    }                                                           \
    void cblas_##X##gemv(                                       \
        const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA,  \
        const int M, const int N,                               \
        const TS alpha, const TV *A, const int lda,             \
        const TV *X, const int incX, const TS beta,             \
        TV *Y, const int incY)                                  \
    {                                                           \
        char aT = transChar(TransA);                            \
        X##gemv_(&aT, &M, &N,                                   \
                 (const TY *)ADDR(alpha), (const TY *)A, &lda,  \
                 (const TY *)X, &incX,                          \
                 (const TY *)ADDR(beta), (TY *)Y, &incY);       \
    }                                                           \
    void cblas_##X##axpy(                                       \
        const int N, const TS alpha,                            \
        const TV *X, const int incX,                            \
        TV *Y, const int incY)                                  \
    {                                                           \
        X##axpy_(&N,                                            \
                 (const TY *)ADDR(alpha),                       \
                 (const TY *)X, &incX,                          \
                 (TY *)Y, &incY);                               \
    }                                                           \
    void cblas_##X##scal(                                       \
        const int N, const TS alpha,                            \
        TV *X, const int incX)                                  \
    {                                                           \
        X##scal_(&N,                                            \
                 (const TY *)ADDR(alpha),                       \
                 (TY *)X, &incX);                               \
    }                                                           \

#define ADDR(val) &val
GEMM_F77(s, float, float, float)
GEMM_F77(d, double, double, double)
#undef ADDR

#define ADDR(val) val
GEMM_F77(c, void *, void, float)
GEMM_F77(z, void *, void, double)
#undef ADDR

#endif
