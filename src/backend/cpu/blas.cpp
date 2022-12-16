/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <blas.hpp>

#ifdef USE_MKL
#include <mkl_cblas.h>
#endif

#include <Array.hpp>
#include <Param.hpp>
#include <common/blas_headers.hpp>
#include <common/cast.hpp>
#include <common/complex.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <copy.hpp>
#include <kernel/dot.hpp>
#include <platform.hpp>
#include <types.hpp>

#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>

#include <algorithm>
#include <type_traits>
#include <vector>

using af::dtype_traits;
using arrayfire::common::cast;
using arrayfire::common::half;
using arrayfire::common::is_complex;
using std::conditional;
using std::vector;

namespace arrayfire {
namespace cpu {

// clang-format off
// Some implementations of BLAS require void* for complex pointers while others
// use float*/double*
//
// Sample cgemm API
// OpenBLAS
// void cblas_cgemm(OPENBLAS_CONST enum CBLAS_ORDER Order,
//                  OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
//                  OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB,
//                  OPENBLAS_CONST blasint M,
//                  OPENBLAS_CONST blasint N,
//                  OPENBLAS_CONST blasint K,
//                  OPENBLAS_CONST float *alpha, OPENBLAS_CONST float *A,
//                  OPENBLAS_CONST blasint lda,
//                  OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb,
//                  OPENBLAS_CONST float *beta,
//                  float *C, OPENBLAS_CONST blasint ldc);
//
// MKL
// void cblas_cgemm(const  CBLAS_LAYOUT Layout,
//                  const CBLAS_TRANSPOSE TransA, const  CBLAS_TRANSPOSE TransB,
//                  const MKL_INT M, const MKL_INT N, const MKL_INT K,
//                  const void *alpha, const void *A, const MKL_INT lda,
//                  const void *B, const MKL_INT ldb, const void *beta,
//                  void *C, const MKL_INT ldc);
// void cblas_cgemm_batch(const  CBLAS_LAYOUT Layout,
//                        const CBLAS_TRANSPOSE* TransA,
//                        const CBLAS_TRANSPOSE* TransB,
//                        const MKL_INT* M, const MKL_INT* N, const MKL_INT* K,
//                        const void *alpha, const void **A, const MKL_INT* lda,
//                        const void **B, const MKL_INT* ldb, const void *beta,
//                        void **C, const MKL_INT* ldc,
//                        const MKL_INT group_count, const MKL_INT* group_size);
//
// atlas cblas
// void cblas_cgemm(const enum CBLAS_ORDER Order,
//                  const enum CBLAS_TRANSPOSE TransA,
//                  const enum CBLAS_TRANSPOSE TransB,
//                  const int M, const int N, const int K,
//                  const void *alpha, const void *A, const int lda,
//                  const void *B, const int ldb, const void *beta,
//                  void *C, const int ldc);
//
// LAPACKE
// void cblas_cgemm(const enum CBLAS_ORDER Order,
//                  const enum CBLAS_TRANSPOSE TransA,
//                  const enum CBLAS_TRANSPOSE TransB,
//                  const int M, const int N, const int K,
//                  const void *alpha, const void *A, const int lda,
//                  const void *B, const int ldb, const void *beta,
//                  void *C, const int ldc);
// clang-format on

template<typename T>
struct blas_base {
    using type =
        typename conditional<is_complex<T>::value && cplx_void_ptr, void,
                             typename dtype_traits<T>::base_type>::type;
};

template<typename T>
using cptr_type =
    typename conditional<is_complex<T>::value,
                         const typename blas_base<T>::type *, const T *>::type;
template<typename T>
using ptr_type = typename conditional<is_complex<T>::value,
                                      typename blas_base<T>::type *, T *>::type;

template<typename T, bool batched = false>
class scale_type {
    const T val;

   public:
    explicit scale_type(const T *val_ptr) : val(*val_ptr) {}
    using api_type = const typename conditional<
        is_complex<T>::value, const typename blas_base<T>::type *,
        const typename conditional<batched, const T *, const T>::type>::type;

    api_type getScale() const {  // NOLINT(readability-const-return-type)
        return val;
    }
};

#define INSTANTIATE_BATCHED(TYPE)              \
    template<>                                 \
    typename scale_type<TYPE, true>::api_type  \
    scale_type<TYPE, true>::getScale() const { \
        return &val;                           \
    }

INSTANTIATE_BATCHED(float);   // NOLINT(readability-const-return-type)
INSTANTIATE_BATCHED(double);  // NOLINT(readability-const-return-type)
#undef INSTANTIATE_BATCHED

#define INSTANTIATE_COMPLEX(TYPE, BATCHED)                                    \
    template<>                                                                \
    scale_type<TYPE, BATCHED>::api_type scale_type<TYPE, BATCHED>::getScale() \
        const {                                                               \
        return reinterpret_cast<const blas_base<TYPE>::type *const>(&val);    \
    }

INSTANTIATE_COMPLEX(cfloat, true);    // NOLINT(readability-const-return-type)
INSTANTIATE_COMPLEX(cfloat, false);   // NOLINT(readability-const-return-type)
INSTANTIATE_COMPLEX(cdouble, true);   // NOLINT(readability-const-return-type)
INSTANTIATE_COMPLEX(cdouble, false);  // NOLINT(readability-const-return-type)
#undef INSTANTIATE_COMPLEX

template<typename T>
using gemm_func_def = void (*)(const CBLAS_ORDER, const CBLAS_TRANSPOSE,
                               const CBLAS_TRANSPOSE, const blasint,
                               const blasint, const blasint,
                               typename scale_type<T>::api_type, cptr_type<T>,
                               const blasint, cptr_type<T>, const blasint,
                               typename scale_type<T>::api_type, ptr_type<T>,
                               const blasint);

template<typename T>
using gemv_func_def = void (*)(const CBLAS_ORDER, const CBLAS_TRANSPOSE,
                               const blasint, const blasint,
                               typename scale_type<T>::api_type, cptr_type<T>,
                               const blasint, cptr_type<T>, const blasint,
                               typename scale_type<T>::api_type, ptr_type<T>,
                               const blasint);

#ifdef USE_MKL
template<typename T>
using gemm_batch_func_def = void (*)(
    const CBLAS_LAYOUT, const CBLAS_TRANSPOSE *, const CBLAS_TRANSPOSE *,
    const MKL_INT *, const MKL_INT *, const MKL_INT *,
    typename scale_type<T, true>::api_type, cptr_type<T> *, const MKL_INT *,
    cptr_type<T> *, const MKL_INT *, typename scale_type<T, true>::api_type,
    ptr_type<T> *, const MKL_INT *, const MKL_INT, const MKL_INT *);
#endif

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

#ifdef USE_MKL
BLAS_FUNC_DEF(gemm_batch)
BLAS_FUNC(gemm_batch, float, s)
BLAS_FUNC(gemm_batch, double, d)
BLAS_FUNC(gemm_batch, cfloat, c)
BLAS_FUNC(gemm_batch, cdouble, z)
#endif

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
    const dim4 oDims  = out.dims();

    using BT  = typename blas_base<T>::type;
    using CBT = const typename blas_base<T>::type;

    auto alpha_ = scale_type<T, false>(alpha);
    auto beta_  = scale_type<T, false>(beta);
#ifdef USE_MKL
    auto alpha_batched = scale_type<T, true>(alpha);
    auto beta_batched  = scale_type<T, true>(beta);
#endif

    auto func = [=](Param<T> output, CParam<T> left, CParam<T> right) {
        dim4 lStrides = left.strides();
        dim4 rStrides = right.strides();
        dim4 oStrides = output.strides();

        if (output.dims().ndims() <= 2) {
            if (right.dims()[bColDim] == 1) {
                dim_t incr =
                    (optRhs == AF_MAT_NONE) ? rStrides[0] : rStrides[1];
                gemv_func<T>()(
                    CblasColMajor, lOpts, lDims[0], lDims[1], alpha_.getScale(),
                    reinterpret_cast<CBT *>(left.get()), lStrides[1],
                    reinterpret_cast<CBT *>(right.get()), incr,
                    beta_.getScale(), reinterpret_cast<BT *>(output.get()),
                    oStrides[0]);
            } else {
                gemm_func<T>()(
                    CblasColMajor, lOpts, rOpts, M, N, K, alpha_.getScale(),
                    reinterpret_cast<CBT *>(left.get()), lStrides[1],
                    reinterpret_cast<CBT *>(right.get()), rStrides[1],
                    beta_.getScale(), reinterpret_cast<BT *>(output.get()),
                    oStrides[1]);
            }
        } else {
            int batchSize = static_cast<int>(oDims[2] * oDims[3]);

            const bool is_l_d2_batched = oDims[2] == lDims[2];
            const bool is_l_d3_batched = oDims[3] == lDims[3];
            const bool is_r_d2_batched = oDims[2] == rDims[2];
            const bool is_r_d3_batched = oDims[3] == rDims[3];

            vector<CBT *> lptrs(batchSize);
            vector<CBT *> rptrs(batchSize);
            vector<BT *> optrs(batchSize);

            for (int n = 0; n < batchSize; n++) {
                ptrdiff_t w = n / oDims[2];
                ptrdiff_t z = n - w * oDims[2];

                ptrdiff_t loff = z * (is_l_d2_batched * lStrides[2]) +
                                 w * (is_l_d3_batched * lStrides[3]);
                ptrdiff_t roff = z * (is_r_d2_batched * rStrides[2]) +
                                 w * (is_r_d3_batched * rStrides[3]);

                lptrs[n] = reinterpret_cast<CBT *>(left.get() + loff);
                rptrs[n] = reinterpret_cast<CBT *>(right.get() + roff);
                optrs[n] = reinterpret_cast<BT *>(
                    output.get() + z * oStrides[2] + w * oStrides[3]);
            }

#ifdef USE_MKL
            // MKL can handle multiple groups of batches
            // However, for ArrayFire's use case, the group_count=1
            const MKL_INT lda = lStrides[1];
            const MKL_INT ldb = rStrides[1];
            const MKL_INT ldc = oStrides[1];

            gemm_batch_func<T>()(CblasColMajor, &lOpts, &rOpts, &M, &N, &K,
                                 alpha_batched.getScale(), lptrs.data(), &lda,
                                 rptrs.data(), &ldb, beta_batched.getScale(),
                                 optrs.data(), &ldc, 1, &batchSize);
#else
            for (int n = 0; n < batchSize; n++) {
                if (rDims[bColDim] == 1) {
                    dim_t incr =
                        (optRhs == AF_MAT_NONE) ? rStrides[0] : rStrides[1];
                    gemv_func<T>()(CblasColMajor, lOpts, lDims[0], lDims[1],
                                   alpha_.getScale(), lptrs[n], lStrides[1],
                                   rptrs[n], incr, beta_.getScale(), optrs[n],
                                   oStrides[0]);
                } else {
                    gemm_func<T>()(CblasColMajor, lOpts, rOpts, M, N, K,
                                   alpha_.getScale(), lptrs[n], lStrides[1],
                                   rptrs[n], rStrides[1], beta_.getScale(),
                                   optrs[n], oStrides[1]);
                }
            }
#endif
        }
    };
    getQueue().enqueue(func, out, lhs, rhs);
}

template<>
void gemm<half>(Array<half> &out, af_mat_prop optLhs, af_mat_prop optRhs,
                const half *alpha, const Array<half> &lhs,
                const Array<half> &rhs, const half *beta) {
    Array<float> outArr    = createValueArray<float>(out.dims(), 0);
    const auto float_alpha = static_cast<float>(*alpha);
    const auto float_beta  = static_cast<float>(*beta);
    gemm<float>(outArr, optLhs, optRhs, &float_alpha, cast<float>(lhs),
                cast<float>(rhs), &float_beta);
    copyArray(out, outArr);
}

template<typename T>
Array<T> dot(const Array<T> &lhs, const Array<T> &rhs, af_mat_prop optLhs,
             af_mat_prop optRhs) {
    Array<T> out = createEmptyArray<T>(af::dim4(1));
    if (optLhs == AF_MAT_CONJ && optRhs == AF_MAT_CONJ) {
        getQueue().enqueue(kernel::dot<T, false, true>, out, lhs, rhs, optLhs,
                           optRhs);
    } else if (optLhs == AF_MAT_CONJ && optRhs == AF_MAT_NONE) {
        getQueue().enqueue(kernel::dot<T, true, false>, out, lhs, rhs, optLhs,
                           optRhs);
    } else if (optLhs == AF_MAT_NONE && optRhs == AF_MAT_CONJ) {
        getQueue().enqueue(kernel::dot<T, true, false>, out, rhs, lhs, optRhs,
                           optLhs);
    } else {
        getQueue().enqueue(kernel::dot<T, false, false>, out, lhs, rhs, optLhs,
                           optRhs);
    }
    return out;
}

template<>
Array<half> dot<half>(const Array<half> &lhs, const Array<half> &rhs,
                      af_mat_prop optLhs, af_mat_prop optRhs) {
    Array<float> out = dot(cast<float>(lhs), cast<float>(rhs), optLhs, optRhs);
    return cast<half>(out);
}

#undef BT
#undef REINTEPRET_CAST

#define INSTANTIATE_GEMM(TYPE)                                               \
    template void gemm<TYPE>(Array<TYPE> & out, af_mat_prop optLhs,          \
                             af_mat_prop optRhs, const TYPE *alphas,         \
                             const Array<TYPE> &lhs, const Array<TYPE> &rhs, \
                             const TYPE *beta)

INSTANTIATE_GEMM(float);
INSTANTIATE_GEMM(cfloat);
INSTANTIATE_GEMM(double);
INSTANTIATE_GEMM(cdouble);

#define INSTANTIATE_DOT(TYPE)                                                  \
    template Array<TYPE> dot<TYPE>(const Array<TYPE> &lhs,                     \
                                   const Array<TYPE> &rhs, af_mat_prop optLhs, \
                                   af_mat_prop optRhs)

INSTANTIATE_DOT(float);
INSTANTIATE_DOT(double);
INSTANTIATE_DOT(cfloat);
INSTANTIATE_DOT(cdouble);

}  // namespace cpu
}  // namespace arrayfire
