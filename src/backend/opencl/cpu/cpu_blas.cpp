/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined(WITH_LINEAR_ALGEBRA)
#include <common/blas_headers.hpp>
#include <common/complex.hpp>
#include <common/err_common.hpp>
#include <cpu/cpu_blas.hpp>
#include <cpu/cpu_helper.hpp>
#include <math.hpp>
#include <traits.hpp>

using arrayfire::common::is_complex;

using std::add_const;
using std::add_pointer;
using std::conditional;
using std::enable_if;
using std::is_floating_point;
using std::remove_const;

namespace arrayfire {
namespace opencl {
namespace cpu {

// Some implementations of BLAS require void* for complex pointers while others
// use float*/double*
//
// Sample cgemm API
// OpenBLAS
// void cblas_cgemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum
// CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB,
//                  OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N,
//                  OPENBLAS_CONST blasint K, OPENBLAS_CONST float *alpha,
//                  OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda,
//                  OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb,
//                  OPENBLAS_CONST float *beta, float *C, OPENBLAS_CONST blasint
//                  ldc);
//
// MKL
// void cblas_cgemm(const  CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
// const  CBLAS_TRANSPOSE TransB,
//                  const MKL_INT M, const MKL_INT N, const MKL_INT K,
//                  const void *alpha, const void *A, const MKL_INT lda,
//                  const void *B, const MKL_INT ldb, const void *beta,
//                  void *C, const MKL_INT ldc);
// atlas cblas
// void cblas_cgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE
// TransA,
//                  const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
//                  const int K, const void *alpha, const void *A, const int
//                  lda, const void *B, const int ldb, const void *beta, void
//                  *C, const int ldc);
//
// LAPACKE
// void cblas_cgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE
// TransA,
//                  const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
//                  const int K, const void *alpha, const void *A, const int
//                  lda, const void *B, const int ldb, const void *beta, void
//                  *C, const int ldc);
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
struct blas_base<
    T, typename enable_if<is_complex<T>::value && cplx_void_ptr>::type> {
    using type = void;
};

template<typename T>
using cptr_type =
    typename conditional<is_complex<T>::value,
                         const typename blas_base<T>::type *, const T *>::type;
template<typename T>
using ptr_type = typename conditional<is_complex<T>::value,
                                      typename blas_base<T>::type *, T *>::type;
template<typename T>
using scale_type =
    typename conditional<is_complex<T>::value,
                         const typename blas_base<T>::type *, const T>::type;

template<typename T>
scale_type<T> getOneScalar(const T *const vals) {
    return vals[0];
}

template<>
scale_type<cfloat> getOneScalar(const cfloat *const vals) {
    return reinterpret_cast<scale_type<cfloat>>(vals);
}

template<>
scale_type<cdouble> getOneScalar(const cdouble *const vals) {
    return reinterpret_cast<scale_type<cdouble>>(vals);
}

template<typename T>
using gemm_func_def = void (*)(const CBLAS_ORDER, const CBLAS_TRANSPOSE,
                               const CBLAS_TRANSPOSE, const blasint,
                               const blasint, const blasint, scale_type<T>,
                               cptr_type<T>, const blasint, cptr_type<T>,
                               const blasint, scale_type<T>, ptr_type<T>,
                               const blasint);

template<typename T>
using gemv_func_def = void (*)(const CBLAS_ORDER, const CBLAS_TRANSPOSE,
                               const blasint, const blasint, scale_type<T>,
                               cptr_type<T>, const blasint, cptr_type<T>,
                               const blasint, scale_type<T>, ptr_type<T>,
                               const blasint);

#define BLAS_FUNC_DEF(FUNC) \
    template<typename T>    \
    FUNC##_func_def<T> FUNC##_func();

#define BLAS_FUNC(FUNC, TYPE, PREFIX)                        \
    template<>                                               \
    FUNC##_func_def<TYPE> FUNC##_func<TYPE>() {              \
        return (FUNC##_func_def<TYPE>)&cblas_##PREFIX##FUNC; \
    }

BLAS_FUNC_DEF(gemm)
BLAS_FUNC(gemm, float, s)
BLAS_FUNC(gemm, double, d)
BLAS_FUNC(gemm, cfloat, c)
BLAS_FUNC(gemm, cdouble, z)

BLAS_FUNC_DEF(gemv)
BLAS_FUNC(gemv, float, s)
BLAS_FUNC(gemv, double, d)
BLAS_FUNC(gemv, cfloat, c)
BLAS_FUNC(gemv, cdouble, z)

template<typename T, int value>
typename enable_if<is_floating_point<T>::value, scale_type<T>>::type
getScale() {
    return T(value);
}

template<typename T, int value>
typename enable_if<is_complex<T>::value, scale_type<T>>::type getScale() {
    thread_local T val = scalar<T>(value);
    return (const typename blas_base<T>::type *)&val;
}

CBLAS_TRANSPOSE
toCblasTranspose(af_mat_prop opt) {
    CBLAS_TRANSPOSE out = CblasNoTrans;
    switch (opt) {
        case AF_MAT_NONE: out = CblasNoTrans; break;
        case AF_MAT_TRANS: out = CblasTrans; break;
        case AF_MAT_CTRANS: out = CblasConjTrans; break;
        default: AF_ERROR("INVALID af_mat_prop", AF_ERR_ARG);
    }
    return out;
}

template<typename T>
void gemm(Array<T> &out, af_mat_prop optLhs, af_mat_prop optRhs, const T *alpha,
          const Array<T> &lhs, const Array<T> &rhs, const T *beta) {
    using BT  = typename blas_base<T>::type;
    using CBT = const typename blas_base<T>::type;

    const CBLAS_TRANSPOSE lOpts = toCblasTranspose(optLhs);
    const CBLAS_TRANSPOSE rOpts = toCblasTranspose(optRhs);

    const int aRowDim = (lOpts == CblasNoTrans) ? 0 : 1;
    const int aColDim = (lOpts == CblasNoTrans) ? 1 : 0;
    const int bColDim = (rOpts == CblasNoTrans) ? 1 : 0;

    const dim4 &lDims = lhs.dims();
    const dim4 &rDims = rhs.dims();
    const int M       = lDims[aRowDim];
    const int N       = rDims[bColDim];
    const int K       = lDims[aColDim];
    const dim4 &oDims = out.dims();

    dim4 lStrides = lhs.strides();
    dim4 rStrides = rhs.strides();
    dim4 oStrides = out.strides();

    int batchSize = oDims[2] * oDims[3];

    bool is_l_d2_batched = (oDims[2] == lDims[2]);
    bool is_l_d3_batched = (oDims[3] == lDims[3]);
    bool is_r_d2_batched = (oDims[2] == rDims[2]);
    bool is_r_d3_batched = (oDims[3] == rDims[3]);

    // get host pointers from mapped memory
    mapped_ptr<T> lPtr = lhs.getMappedPtr(CL_MAP_READ);
    mapped_ptr<T> rPtr = rhs.getMappedPtr(CL_MAP_READ);
    mapped_ptr<T> oPtr = out.getMappedPtr(CL_MAP_READ | CL_MAP_WRITE);

    for (int n = 0; n < batchSize; ++n) {
        int w = n / oDims[2];
        int z = n - w * oDims[2];

        int loff = z * (is_l_d2_batched * lStrides[2]) +
                   w * (is_l_d3_batched * lStrides[3]);
        int roff = z * (is_r_d2_batched * rStrides[2]) +
                   w * (is_r_d3_batched * rStrides[3]);

        CBT *lptr = reinterpret_cast<CBT *>(lPtr.get() + loff);
        CBT *rptr = reinterpret_cast<CBT *>(rPtr.get() + roff);
        BT *optr  = reinterpret_cast<BT *>(oPtr.get() + z * oStrides[2] +
                                          w * oStrides[3]);

        if (rDims[bColDim] == 1) {
            dim_t incr = (rOpts == CblasNoTrans) ? rStrides[0] : rStrides[1];
            gemv_func<T>()(CblasColMajor, lOpts, lDims[0], lDims[1],
                           getOneScalar<T>(alpha), lptr, lStrides[1], rptr,
                           incr, getOneScalar<T>(beta), optr, 1);
        } else {
            gemm_func<T>()(CblasColMajor, lOpts, rOpts, M, N, K,
                           getOneScalar<T>(alpha), lptr, lStrides[1], rptr,
                           rStrides[1], getOneScalar<T>(beta), optr,
                           oStrides[1]);
        }
    }
}

#define INSTANTIATE_GEMM(TYPE)                                               \
    template void gemm<TYPE>(Array<TYPE> & out, af_mat_prop optLhs,          \
                             af_mat_prop optRhs, const TYPE *alpha,          \
                             const Array<TYPE> &lhs, const Array<TYPE> &rhs, \
                             const TYPE *beta);

INSTANTIATE_GEMM(float)
INSTANTIATE_GEMM(cfloat)
INSTANTIATE_GEMM(double)
INSTANTIATE_GEMM(cdouble)

}  // namespace cpu
}  // namespace opencl
}  // namespace arrayfire
#endif  // WITH_LINEAR_ALGEBRA
