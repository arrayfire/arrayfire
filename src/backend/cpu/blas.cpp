/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <blas.hpp>
#include <af/dim4.hpp>
#include <cassert>
#include <common/err_common.hpp>
#include <kernel/dot.hpp>
#include <platform.hpp>
#include <queue.hpp>

#include <common/blas_headers.hpp>

namespace cpu
{

using std::add_const;
using std::add_pointer;
using std::enable_if;
using std::is_floating_point;
using std::remove_const;
using std::conditional;

// Some implementations of BLAS require void* for complex pointers while others use float*/double*
//
// Sample cgemm API
// OpenBLAS
// void cblas_cgemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB,
//                  OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
//                  OPENBLAS_CONST float *alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda,
//                  OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float *beta,
//                  float *C, OPENBLAS_CONST blasint ldc);
//
// MKL
// void cblas_cgemm(const  CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA, const  CBLAS_TRANSPOSE TransB,
//                  const MKL_INT M, const MKL_INT N, const MKL_INT K,
//                  const void *alpha, const void *A, const MKL_INT lda,
//                  const void *B, const MKL_INT ldb, const void *beta,
//                  void *C, const MKL_INT ldc);
// atlas cblas
// void cblas_cgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
//                  const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
//                  const void *alpha, const void *A, const int lda,
//                  const void *B, const int ldb, const void *beta,
//                  void *C, const int ldc);
//
// LAPACKE
// void cblas_cgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
//                  const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
//                  const void *alpha, const void *A, const int lda,
//                  const void *B, const int ldb, const void *beta,
//                  void *C, const int ldc);

template<typename T>
struct blas_base {
    using type = typename conditional<is_complex<T>::value && cplx_void_ptr,
                                      void,
                                      typename dtype_traits<T>::base_type>::type;
};

template<typename T>
using cptr_type     =   typename conditional<   is_complex<T>::value,
                                                const typename blas_base<T>::type *,
                                                const T*>::type;
template<typename T>
using ptr_type     =    typename conditional<   is_complex<T>::value,
                                                typename blas_base<T>::type *,
                                                T*>::type;
template<typename T>
using scale_type   =    typename conditional<   is_complex<T>::value,
                                                const typename blas_base<T>::type *,
                                                const T>::type;

template<typename T>
using gemm_func_def = void (*)( const CBLAS_ORDER, const CBLAS_TRANSPOSE, const CBLAS_TRANSPOSE,
                                const blasint, const blasint, const blasint,
                                scale_type<T>, cptr_type<T>, const blasint,
                                cptr_type<T>, const blasint,
                                scale_type<T>, ptr_type<T>, const blasint);

template<typename T>
using gemv_func_def = void (*)( const CBLAS_ORDER, const CBLAS_TRANSPOSE,
                                const blasint, const blasint,
                                scale_type<T>, cptr_type<T>, const blasint,
                                cptr_type<T>, const blasint,
                                scale_type<T>, ptr_type<T>, const blasint);

#define BLAS_FUNC_DEF( FUNC )                           \
template<typename T> FUNC##_func_def<T> FUNC##_func();

#define BLAS_FUNC( FUNC, TYPE, PREFIX )                 \
  template<> FUNC##_func_def<TYPE> FUNC##_func<TYPE>()  \
{ return &cblas_##PREFIX##FUNC; }

BLAS_FUNC_DEF( gemm )
BLAS_FUNC(gemm , float   , s)
BLAS_FUNC(gemm , double  , d)
BLAS_FUNC(gemm , cfloat  , c)
BLAS_FUNC(gemm , cdouble , z)

BLAS_FUNC_DEF(gemv)
BLAS_FUNC(gemv , float   , s)
BLAS_FUNC(gemv , double  , d)
BLAS_FUNC(gemv , cfloat  , c)
BLAS_FUNC(gemv , cdouble , z)

template<typename T, int value>
typename enable_if<is_floating_point<T>::value, scale_type<T>>::type
getScale() { return T(value); }

template<typename T, int value>
typename enable_if<is_complex<T>::value, scale_type<T>>::type
getScale()
{
    static T val(value);
    return (const typename blas_base<T>::type *)&val;
}

CBLAS_TRANSPOSE
toCblasTranspose(af_mat_prop opt)
{
    CBLAS_TRANSPOSE out = CblasNoTrans;
    switch(opt) {
        case AF_MAT_NONE        : out = CblasNoTrans;   break;
        case AF_MAT_TRANS       : out = CblasTrans;     break;
        case AF_MAT_CTRANS      : out = CblasConjTrans; break;
        default                 : AF_ERROR("INVALID af_mat_prop", AF_ERR_ARG);
    }
    return out;
}

template<typename T>
Array<T> matmul(const Array<T> &lhs, const Array<T> &rhs,
                af_mat_prop optLhs, af_mat_prop optRhs)
{
    lhs.eval();
    rhs.eval();

    CBLAS_TRANSPOSE lOpts = toCblasTranspose(optLhs);
    CBLAS_TRANSPOSE rOpts = toCblasTranspose(optRhs);

    int aRowDim = (lOpts == CblasNoTrans) ? 0 : 1;
    int aColDim = (lOpts == CblasNoTrans) ? 1 : 0;
    int bColDim = (rOpts == CblasNoTrans) ? 1 : 0;

    dim4 lDims = lhs.dims();
    dim4 rDims = rhs.dims();
    int M = lDims[aRowDim];
    int N = rDims[bColDim];
    int K = lDims[aColDim];

    using BT  =       typename blas_base<T>::type;
    using CBT = const typename blas_base<T>::type;

    dim_t d2 = std::max(lDims[2], rDims[2]);
    dim_t d3 = std::max(lDims[3], rDims[3]);
    dim4 oDims = af::dim4(M, N, d2, d3);
    Array<T> out = createEmptyArray<T>(oDims);

    auto func = [=] (Param<T> output, CParam<T> left, CParam<T> right) {
        auto alpha = getScale<T, 1>();
        auto beta  = getScale<T, 0>();

        dim4 lStrides = left.strides();
        dim4 rStrides = right.strides();
        dim4 oStrides = output.strides();

        int batchSize = oDims[2] * oDims[3];

        bool is_l_d2_batched = oDims[2] == lDims[2];
        bool is_l_d3_batched = oDims[3] == lDims[3];
        bool is_r_d2_batched = oDims[2] == rDims[2];
        bool is_r_d3_batched = oDims[3] == rDims[3];

        for (int n = 0; n < batchSize; n++) {
            int w = n / rDims[2];
            int z = n - w * rDims[2];

            int loff = z * (is_l_d2_batched * lStrides[2]) + w * (is_l_d3_batched * lStrides[3]);
            int roff = z * (is_r_d2_batched * rStrides[2]) + w * (is_r_d3_batched * rStrides[3]);

            CBT *lptr = reinterpret_cast<CBT*>(left.get() + loff);
            CBT *rptr = reinterpret_cast<CBT*>(right.get() + roff);
            BT *optr = reinterpret_cast<BT*>(output.get() + z * oStrides[2] + w * oStrides[3]);

            if(rDims[bColDim] == 1) {
                dim_t incr = (optRhs == AF_MAT_NONE) ? rStrides[0] : rStrides[1];
                gemv_func<T>()(
                    CblasColMajor, lOpts,
                    lDims[0], lDims[1],
                    alpha,
                    lptr, lStrides[1],
                    rptr, incr,
                    beta,
                    optr, 1);
            } else {
                gemm_func<T>()(
                    CblasColMajor, lOpts, rOpts,
                    M, N, K,
                    alpha,
                    lptr, lStrides[1],
                    rptr, rStrides[1],
                    beta,
                    optr, output.dims(0));
            }
        }
    };
    getQueue().enqueue(func, out, lhs, rhs);

    return out;
}

template<typename T>
Array<T> dot(const Array<T> &lhs, const Array<T> &rhs,
             af_mat_prop optLhs, af_mat_prop optRhs)
{
    lhs.eval();
    rhs.eval();

    Array<T> out = createEmptyArray<T>(af::dim4(1));
    if(optLhs == AF_MAT_CONJ && optRhs == AF_MAT_CONJ) {
        getQueue().enqueue(kernel::dot<T, false, true>, out, lhs, rhs, optLhs, optRhs);
    } else if (optLhs == AF_MAT_CONJ && optRhs == AF_MAT_NONE) {
        getQueue().enqueue(kernel::dot<T, true, false>,out, lhs, rhs, optLhs, optRhs);
    } else if (optLhs == AF_MAT_NONE && optRhs == AF_MAT_CONJ) {
        getQueue().enqueue(kernel::dot<T, true, false>,out, rhs, lhs, optRhs, optLhs);
    } else {
        getQueue().enqueue(kernel::dot<T, false, false>,out, lhs, rhs, optLhs, optRhs);
    }
    return out;
}

#undef BT
#undef REINTEPRET_CAST

#define INSTANTIATE_BLAS(TYPE)                                                          \
    template Array<TYPE> matmul<TYPE>(const Array<TYPE> &lhs, const Array<TYPE> &rhs,   \
                                      af_mat_prop optLhs, af_mat_prop optRhs);

INSTANTIATE_BLAS(float)
INSTANTIATE_BLAS(cfloat)
INSTANTIATE_BLAS(double)
INSTANTIATE_BLAS(cdouble)

#define INSTANTIATE_DOT(TYPE)                                                               \
    template Array<TYPE> dot<TYPE>(const Array<TYPE> &lhs, const Array<TYPE> &rhs,          \
                                   af_mat_prop optLhs, af_mat_prop optRhs);

INSTANTIATE_DOT(float)
INSTANTIATE_DOT(double)
INSTANTIATE_DOT(cfloat)
INSTANTIATE_DOT(cdouble)

}
