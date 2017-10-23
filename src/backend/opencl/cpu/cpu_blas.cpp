/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined(WITH_LINEAR_ALGEBRA)
#include <cpu/cpu_helper.hpp>
#include <cpu/cpu_blas.hpp>
#include <math.hpp>
#include <common/blas_headers.hpp>

namespace opencl
{
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
#if defined(IS_OPENBLAS)
    static const bool cplx_void_ptr = false;
#else
    static const bool cplx_void_ptr = true;
#endif

template<typename T, class Enable = void>
struct blas_base {
    using type = typename dtype_traits<T>::base_type;
};

template<typename T>
struct blas_base <T, typename enable_if<is_complex<T>::value && cplx_void_ptr>::type> {
    using type = void;
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
    thread_local T val = scalar<T>(value);
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

    //FIXME: Leaks on errors.
    Array<T> out = createValueArray<T>(af::dim4(M, N, 1, 1), scalar<T>(0));
    auto alpha = getScale<T, 1>();
    auto beta  = getScale<T, 0>();

    dim4 lStrides = lhs.strides();
    dim4 rStrides = rhs.strides();
    using BT  =       typename blas_base<T>::type;

    // get host pointers from mapped memory
    auto lPtr = lhs.getMappedPtr();
    auto rPtr = rhs.getMappedPtr();
    auto oPtr = out.getMappedPtr();

    if(rDims[bColDim] == 1) {
        N = lDims[aColDim];
        gemv_func<T>()(
            CblasColMajor, lOpts,
            lDims[0], lDims[1],
            alpha,
            (BT*)lPtr.get(), lStrides[1],
            (BT*)rPtr.get(), rStrides[0],
            beta,
            (BT*)oPtr.get(), 1);
    } else {
        gemm_func<T>()(
            CblasColMajor, lOpts, rOpts,
            M, N, K,
            alpha,
            (BT*)lPtr.get(), lStrides[1],
            (BT*)rPtr.get(), rStrides[1],
            beta,
            (BT*)oPtr.get(), out.dims()[0]);
    }

    return out;
}

#define INSTANTIATE_BLAS(TYPE)                                                          \
    template Array<TYPE> matmul<TYPE>(const Array<TYPE> &lhs, const Array<TYPE> &rhs,   \
                                      af_mat_prop optLhs, af_mat_prop optRhs);

INSTANTIATE_BLAS(float)
INSTANTIATE_BLAS(cfloat)
INSTANTIATE_BLAS(double)
INSTANTIATE_BLAS(cdouble)

}
}
#endif  // WITH_LINEAR_ALGEBRA
