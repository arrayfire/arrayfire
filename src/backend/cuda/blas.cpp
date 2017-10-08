/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <blas.hpp>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <platform.hpp>

#include <stdexcept>
#include <string>
#include <cassert>
#include <math.hpp>
#include <common/err_common.hpp>
#include <cublas.hpp>
#include <arith.hpp>
#include <reduce.hpp>
#include <complex.hpp>

namespace cuda
{

cublasOperation_t
toCblasTranspose(af_mat_prop opt)
{
    cublasOperation_t out = CUBLAS_OP_N;
    switch(opt) {
        case AF_MAT_NONE        : out = CUBLAS_OP_N;    break;
        case AF_MAT_TRANS           : out = CUBLAS_OP_T;    break;
        case AF_MAT_CTRANS : out = CUBLAS_OP_C;    break;
        default                     : AF_ERROR("INVALID af_mat_prop", AF_ERR_ARG);
    }
    return out;
}

template<typename T>
struct gemm_func_def_t
{
    typedef cublasStatus_t (*gemm_func_def)(    cublasHandle_t,
                                                cublasOperation_t, cublasOperation_t,
                                                int, int, int,
                                                const T *,  const T *, int,
                                                            const T *, int,
                                                const T *,        T *, int);
};

template<typename T>
struct gemv_func_def_t
{
    typedef cublasStatus_t (*gemv_func_def)(    cublasHandle_t,
                                                cublasOperation_t,
                                                int, int,
                                                const T *,  const T *, int,
                                                            const T *, int,
                                                const T *,        T *, int);
};

template<typename T>
struct trsm_func_def_t
{
    typedef cublasStatus_t (*trsm_func_def)(    cublasHandle_t,
                                                cublasSideMode_t,
                                                cublasFillMode_t,
                                                cublasOperation_t,
                                                cublasDiagType_t,
                                                int, int,
                                                const T *,
                                                const T *, int,
                                                T *, int);
};

#define BLAS_FUNC_DEF( FUNC )                       \
template<typename T>                                \
typename FUNC##_func_def_t<T>::FUNC##_func_def      \
FUNC##_func();

#define BLAS_FUNC( FUNC, TYPE, PREFIX )         \
template<> typename FUNC##_func_def_t<TYPE>::FUNC##_func_def       FUNC##_func<TYPE>()  { return (FUNC##_func_def_t<TYPE>::FUNC##_func_def)&cublas##PREFIX##FUNC; }

BLAS_FUNC_DEF(gemm)
BLAS_FUNC(gemm, float,  S)
BLAS_FUNC(gemm, cfloat, C)
BLAS_FUNC(gemm, double, D)
BLAS_FUNC(gemm, cdouble,Z)

BLAS_FUNC_DEF(gemv)
BLAS_FUNC(gemv, float,  S)
BLAS_FUNC(gemv, cfloat, C)
BLAS_FUNC(gemv, double, D)
BLAS_FUNC(gemv, cdouble,Z)

BLAS_FUNC_DEF(trsm)
BLAS_FUNC(trsm, float,  S)
BLAS_FUNC(trsm, cfloat, C)
BLAS_FUNC(trsm, double, D)
BLAS_FUNC(trsm, cdouble,Z)

#undef BLAS_FUNC
#undef BLAS_FUNC_DEF

template<typename T, bool conjugate>
struct dot_func_def_t
{
    typedef cublasStatus_t (*dot_func_def)(    cublasHandle_t,
                                                int,
                                                const T *,  int,
                                                const T *,  int,
                                                T *);
};

#define BLAS_FUNC_DEF( FUNC )                                   \
template<typename T, bool conjugate>                            \
typename FUNC##_func_def_t<T, conjugate>::FUNC##_func_def       \
FUNC##_func();

#define BLAS_FUNC( FUNC, TYPE, CONJUGATE, PREFIX )                           \
template<> typename FUNC##_func_def_t<TYPE, CONJUGATE>::FUNC##_func_def      \
FUNC##_func<TYPE, CONJUGATE>()  { return (FUNC##_func_def_t<TYPE, CONJUGATE>::FUNC##_func_def)&cublas##PREFIX##FUNC; }

BLAS_FUNC_DEF(dot)
BLAS_FUNC(dot, float,  true,  S)
BLAS_FUNC(dot, double, true,  D)
BLAS_FUNC(dot, float,  false, S)
BLAS_FUNC(dot, double, false, D)

#undef BLAS_FUNC

#define BLAS_FUNC( FUNC, TYPE, CONJUGATE, PREFIX, SUFFIX)                \
template<> typename FUNC##_func_def_t<TYPE, CONJUGATE>::FUNC##_func_def  \
FUNC##_func<TYPE, CONJUGATE>()  { return (FUNC##_func_def_t<TYPE, CONJUGATE>::FUNC##_func_def)&cublas##PREFIX##FUNC##SUFFIX; }

BLAS_FUNC_DEF(dot)
BLAS_FUNC(dot, cfloat,  true , C, c)
BLAS_FUNC(dot, cdouble, true , Z, c)
BLAS_FUNC(dot, cfloat,  false, C, u)
BLAS_FUNC(dot, cdouble, false, Z, u)

#undef BLAS_FUNC
#undef BLAS_FUNC_DEF

using namespace std;

template<typename T>
Array<T> matmul(const Array<T> &lhs, const Array<T> &rhs,
                af_mat_prop optLhs, af_mat_prop optRhs)
{
    cublasOperation_t lOpts = toCblasTranspose(optLhs);
    cublasOperation_t rOpts = toCblasTranspose(optRhs);

    int aRowDim = (lOpts == CUBLAS_OP_N) ? 0 : 1;
    int aColDim = (lOpts == CUBLAS_OP_N) ? 1 : 0;
    int bColDim = (rOpts == CUBLAS_OP_N) ? 1 : 0;

    dim4 lDims = lhs.dims();
    dim4 rDims = rhs.dims();
    int M = lDims[aRowDim];
    int N = rDims[bColDim];
    int K = lDims[aColDim];

    Array<T> out = createEmptyArray<T>(af::dim4(M, N, 1, 1));
    T alpha = scalar<T>(1);
    T beta  = scalar<T>(0);

    dim4 lStrides = lhs.strides();
    dim4 rStrides = rhs.strides();
    if(rDims[bColDim] == 1) {
        N = lDims[aColDim];
        dim_t incr = (rOpts == CUBLAS_OP_N) ? rStrides[0] : rStrides[1];
        CUBLAS_CHECK(gemv_func<T>()(
                         blasHandle(),
                         lOpts,
                         lDims[0],
                         lDims[1],
                         &alpha,
                         lhs.get(), lStrides[1],
                         rhs.get(), incr,
                         &beta,
                         out.get(), 1));
    } else {
        CUBLAS_CHECK(gemm_func<T>()(
                         blasHandle(),
                         lOpts,
                         rOpts,
                         M, N, K,
                         &alpha,
                         lhs.get(), lStrides[1],
                         rhs.get(), rStrides[1],
                         &beta,
                         out.get(),
                         out.dims()[0]));
    }

    return out;

}

template<typename T>
Array<T> dot(const Array<T> &lhs, const Array<T> &rhs,
             af_mat_prop optLhs, af_mat_prop optRhs)
{
    const Array<T> lhs_ = (optLhs == AF_MAT_NONE ? lhs : conj<T>(lhs));
    const Array<T> rhs_ = (optRhs == AF_MAT_NONE ? rhs : conj<T>(rhs));

    const Array<T> temp = arithOp<T, af_mul_t>(lhs_, rhs_, lhs_.dims());
    return reduce<af_add_t, T, T>(temp, 0, false, 0);
}

template<typename T>
void trsm(const Array<T> &lhs, Array<T> &rhs, af_mat_prop trans,
          bool is_upper, bool is_left, bool is_unit)
{
    //dim4 lDims = lhs.dims();
    dim4 rDims = rhs.dims();
    int M = rDims[0];
    int N = rDims[1];

    T alpha = scalar<T>(1);

    dim4 lStrides = lhs.strides();
    dim4 rStrides = rhs.strides();

    CUBLAS_CHECK(trsm_func<T>()(
                     blasHandle(),
                     is_left  ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
                     is_upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER,
                     toCblasTranspose(trans),
                     is_unit  ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
                     M, N,
                     &alpha,
                     lhs.get(), lStrides[1],
                     rhs.get(), rStrides[1]));
}


#define INSTANTIATE_BLAS(TYPE)                                                          \
    template Array<TYPE> matmul<TYPE>(const Array<TYPE> &lhs, const Array<TYPE> &rhs,   \
                                      af_mat_prop optLhs, af_mat_prop optRhs);

INSTANTIATE_BLAS(float)
INSTANTIATE_BLAS(cfloat)
INSTANTIATE_BLAS(double)
INSTANTIATE_BLAS(cdouble)

#define INSTANTIATE_DOT(TYPE)                                                           \
    template Array<TYPE> dot<TYPE>(const Array<TYPE> &lhs, const Array<TYPE> &rhs,      \
                                   af_mat_prop optLhs, af_mat_prop optRhs);

INSTANTIATE_DOT(float)
INSTANTIATE_DOT(double)
INSTANTIATE_DOT(cfloat)
INSTANTIATE_DOT(cdouble)

#define INSTANTIATE_TRSM(TYPE)                                                          \
    template void trsm<TYPE>(const Array<TYPE> &lhs, Array<TYPE> &rhs,                  \
                             af_mat_prop trans, bool is_upper, bool is_left, bool is_unit);

INSTANTIATE_TRSM(float)
INSTANTIATE_TRSM(cfloat)
INSTANTIATE_TRSM(double)
INSTANTIATE_TRSM(cdouble)

}
