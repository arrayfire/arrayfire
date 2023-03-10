/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <blas.hpp>

#include <Array.hpp>
#include <arith.hpp>
#include <common/kernel_type.hpp>
#include <common/half.hpp>
#include <common/traits.hpp>
#include <complex.hpp>
#include <err_oneapi.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <reduce.hpp>
#include <transpose.hpp>

#include <sycl/sycl.hpp>
#include "oneapi/mkl/blas.hpp"

#include <complex>
#include <vector>

using arrayfire::common::half;

// Converts an af_mat_prop options to a transpose type for mkl
static oneapi::mkl::transpose
toBlasTranspose(af_mat_prop opt) {
    switch (opt) {
        case   AF_MAT_NONE: return oneapi::mkl::transpose::nontrans;
        case  AF_MAT_TRANS: return oneapi::mkl::transpose::trans;
        case AF_MAT_CTRANS: return oneapi::mkl::transpose::conjtrans;
        default: AF_ERROR("INVALID af_mat_prop", AF_ERR_ARG);
    }
}

template<typename T>
static void gemvDispatch(sycl::queue queue,
                         oneapi::mkl::transpose lOpts,
                         int M, int N, const T *alpha,
                         const arrayfire::oneapi::Array<T> &lhs, dim_t lStride,
                         const arrayfire::oneapi::Array<T> &x, dim_t incx,
                         const T *beta, arrayfire::oneapi::Array<T> &out,
                         dim_t oInc) {
    try {
        sycl::buffer<T> *lhsPtr = const_cast<sycl::buffer<T>*>(lhs.get());
        sycl::buffer<T> *xPtr   = const_cast<sycl::buffer<T>*>(x.get());
        sycl::buffer<T> *oPtr   = const_cast<sycl::buffer<T>*>(out.get());
        oneapi::mkl::blas::gemv(queue, lOpts, M, N, 
            *alpha, *lhsPtr, lStride, *xPtr, incx, 
            *beta, *oPtr, oInc);
    }
    catch(sycl::exception const& e) {
        AF_ERROR(
            "Synchronous SYCL exception during GEMV invocation: "
            + std::string(e.what()),
            AF_ERR_RUNTIME);
    }
}

template<typename T>
static void gemmDispatch(sycl::queue queue,
                         oneapi::mkl::transpose lOpts,
                         oneapi::mkl::transpose rOpts,
                         int M, int N, int K, const T *alpha,
                         const arrayfire::oneapi::Array<T> &lhs, dim_t lStride,
                         const arrayfire::oneapi::Array<T> &rhs, dim_t rStride,
                         const T *beta, arrayfire::oneapi::Array<T> &out,
                         dim_t oleading) {
    try {
        sycl::buffer<T> *lhsPtr = const_cast<sycl::buffer<T>*>(lhs.get());
        sycl::buffer<T> *rhsPtr = const_cast<sycl::buffer<T>*>(rhs.get());
        oneapi::mkl::blas::gemm(queue, lOpts, rOpts, M, N, K, 
            *alpha, *lhsPtr, lStride, *rhsPtr, rStride, *beta,
            *out.get(), oleading);
    }
    catch(sycl::exception const& e) {
        AF_ERROR(
            "Synchronous SYCL exception during GEMM invocation: "
            + std::string(e.what()),
            AF_ERR_RUNTIME);
    }
}

namespace arrayfire {
namespace oneapi {

void initBlas() { /*gpu_blas_init();*/
}

void deInitBlas() { /*gpu_blas_deinit();*/
}

template<>
void gemm(Array<half> &out, af_mat_prop optLhs, af_mat_prop optRhs, const half *alpha,
          const Array<half> &lhs, const Array<half> &rhs, const half *beta) {
    ONEAPI_NOT_SUPPORTED("");
}

template<typename T>
void gemm(Array<T> &out, af_mat_prop optLhs, af_mat_prop optRhs, const T *alpha,
          const Array<T> &lhs, const Array<T> &rhs, const T *beta) {
    const auto lOpts = toBlasTranspose(optLhs);
    const auto rOpts = toBlasTranspose(optRhs);

    const auto aRowDim = (optLhs == AF_MAT_NONE) ? 0 : 1;
    const auto aColDim = (optLhs == AF_MAT_NONE) ? 1 : 0;
    const auto bColDim = (optRhs == AF_MAT_NONE) ? 1 : 0;

    const dim4 &lDims = lhs.dims();
    const dim4 &rDims = rhs.dims();
    const int M       = lDims[aRowDim];
    const int N       = rDims[bColDim];
    const int K       = lDims[aColDim];
    const dim4 oDims  = out.dims();

    const dim4 &lStrides = lhs.strides();
    const dim4 &rStrides = rhs.strides();
    const dim4 oStrides  = out.strides();

    if (oDims.ndims() <= 2) { // if non-batched
        if (rhs.dims()[bColDim] == 1) {
            dim_t incr =
                (optRhs == AF_MAT_NONE) ? rStrides[0] : rStrides[1];
            printf("[gemv]\n");
            gemvDispatch<T>(getQueue(),
                lOpts, M, N, alpha, lhs, lStrides[1],
                rhs, incr, beta, out, oStrides[0]);
        } else {
            printf("[gemm]\n");
            gemmDispatch<T>(getQueue(), lOpts, rOpts, M, N, K, alpha,
                                        lhs, lStrides[1], rhs, rStrides[1], beta,
                                        out, oStrides[1]);
        }
    } else { // if batched
        int batchSize = static_cast<int>(oDims[2] * oDims[3]);

        const bool is_l_d2_batched = oDims[2] == lDims[2];
        const bool is_l_d3_batched = oDims[3] == lDims[3];
        const bool is_r_d2_batched = oDims[2] == rDims[2];
        const bool is_r_d3_batched = oDims[3] == rDims[3];

        std::vector<sycl::buffer<T>> lptrs;
        std::vector<sycl::buffer<T>> rptrs;
        std::vector<sycl::buffer<T>> optrs;

        lptrs.reserve(batchSize);
        rptrs.reserve(batchSize);
        optrs.reserve(batchSize);

        for (int n = 0; n < batchSize; n++) {
            ptrdiff_t w = n / oDims[2];
            ptrdiff_t z = n - w * oDims[2];

            ptrdiff_t loff = z * (is_l_d2_batched * lStrides[2]) +
                                w * (is_l_d3_batched * lStrides[3]);
            ptrdiff_t roff = z * (is_r_d2_batched * rStrides[2]) +
                                w * (is_r_d3_batched * rStrides[3]);
            ptrdiff_t zoff = z * oStrides[2] + w * oStrides[3];

            lptrs.emplace_back(*(const_cast<sycl::buffer<T>*>(lhs.get())), sycl::id<1>(loff), sycl::range<1>(M * K));
            rptrs.emplace_back(*(const_cast<sycl::buffer<T>*>(rhs.get())), sycl::id<1>(roff), sycl::range<1>(M * K));
            optrs.emplace_back(*(const_cast<sycl::buffer<T>*>(out.get())), sycl::id<1>(zoff), sycl::range<1>(M * K));
        }

        for (int n = 0; n < batchSize; n++) {
            if (rDims[bColDim] == 1) {
                try {
                    printf("BATCH gemv\n");
                    dim_t incr =
                        (optRhs == AF_MAT_NONE) ? rStrides[0] : rStrides[1];
                    ::oneapi::mkl::blas::gemv(getQueue(), lOpts, M, N, 
                        *alpha, lptrs[n], lStrides[1], rptrs[n], incr, 
                        *beta, optrs[n], oStrides[0]);
                }
                catch(sycl::exception const& e) {
                    AF_ERROR(
                        "Synchronous SYCL exception during batched GEMV invocation: "
                        + std::string(e.what()),
                        AF_ERR_RUNTIME);
                }
            } else {
                try {
                    printf("BATCH gemm\n");
                    ::oneapi::mkl::blas::gemm(getQueue(), lOpts, rOpts, M, N, K, 
                        *alpha, lptrs[n], lStrides[1], rptrs[n], rStrides[1], *beta,
                        optrs[n], oStrides[1]);
                }
                catch(sycl::exception const& e) {
                    AF_ERROR(
                        "Synchronous SYCL exception during batched GEMM invocation: "
                        + std::string(e.what()),
                        AF_ERR_RUNTIME);
                }
            }
        }
    }
    getQueue().wait_and_throw();
}


template<typename T>
Array<T> dot(const Array<T> &lhs, const Array<T> &rhs, af_mat_prop optLhs,
             af_mat_prop optRhs) {
    auto lhs_ = (optLhs == AF_MAT_NONE ? lhs : conj<T>(lhs));
    auto rhs_ = (optRhs == AF_MAT_NONE ? rhs : conj<T>(rhs));
    auto temp = arithOp<T, af_mul_t>(lhs_, rhs_, lhs_.dims());
    return reduce<af_add_t, T, T>(temp, 0, false, 0);
}

#define INSTANTIATE_GEMM(TYPE)                                              \
    template void gemm<TYPE>(Array<TYPE> & out, af_mat_prop optLhs,          \
                             af_mat_prop optRhs, const TYPE *alpha,          \
                             const Array<TYPE> &lhs, const Array<TYPE> &rhs, \
                             const TYPE *beta);

INSTANTIATE_GEMM(float)
INSTANTIATE_GEMM(cfloat)
INSTANTIATE_GEMM(double)
INSTANTIATE_GEMM(cdouble)
//INSTANTIATE_GEMM(half)

#define INSTANTIATE_DOT(TYPE)                                                  \
    template Array<TYPE> dot<TYPE>(const Array<TYPE> &lhs,                     \
                                   const Array<TYPE> &rhs, af_mat_prop optLhs, \
                                   af_mat_prop optRhs);

INSTANTIATE_DOT(float)
INSTANTIATE_DOT(double)
INSTANTIATE_DOT(cfloat)
INSTANTIATE_DOT(cdouble)
INSTANTIATE_DOT(half)

}  // namespace oneapi
}  // namespace arrayfire

