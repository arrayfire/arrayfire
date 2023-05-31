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
#include <common/half.hpp>
#include <common/kernel_type.hpp>
#include <common/traits.hpp>
#include <complex.hpp>
#include <debug_oneapi.hpp>
#include <err_oneapi.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <reduce.hpp>
#include <transpose.hpp>
#include <types.hpp>

#include <sycl/sycl.hpp>

#include <oneapi/mkl/blas.hpp>

#include <complex>
#include <vector>

using arrayfire::common::half;

// Converts an af_mat_prop options to a transpose type for mkl
static oneapi::mkl::transpose toBlasTranspose(af_mat_prop opt) {
    switch (opt) {
        case AF_MAT_NONE: return oneapi::mkl::transpose::nontrans;
        case AF_MAT_TRANS: return oneapi::mkl::transpose::trans;
        case AF_MAT_CTRANS: return oneapi::mkl::transpose::conjtrans;
        default: AF_ERROR("INVALID af_mat_prop", AF_ERR_ARG);
    }
}

template<typename T>
static void gemvDispatch(sycl::queue queue, oneapi::mkl::transpose lOpts,
                         oneapi::mkl::transpose rOpts, int M, int N,
                         const T *alpha, const arrayfire::oneapi::Array<T> &lhs,
                         dim_t lStride, const arrayfire::oneapi::Array<T> &x,
                         dim_t incx, const T *beta,
                         arrayfire::oneapi::Array<T> &out, dim_t oInc) {
    using Dt                   = arrayfire::oneapi::data_t<T>;
    const af::dim4 lStrides    = lhs.strides();
    const af::dim4 xStrides    = x.strides();
    const af::dim4 oStrides    = out.strides();
    sycl::buffer<Dt, 1> lhsBuf = lhs.template getBufferWithOffset<Dt>();
    sycl::buffer<Dt, 1> xBuf   = x.template getBufferWithOffset<Dt>();
    sycl::buffer<Dt, 1> outBuf = out.template getBufferWithOffset<Dt>();
    if constexpr (!std::is_same_v<T, arrayfire::common::half>) {
        ::oneapi::mkl::blas::gemv(queue, lOpts, (int64_t)M, (int64_t)N,
                                  (T)*alpha, lhsBuf, (int64_t)lStride, xBuf,
                                  (int64_t)incx, (T)*beta, outBuf,
                                  (int64_t)oInc);
    }
}

template<typename T>
static void gemmDispatch(sycl::queue queue, oneapi::mkl::transpose lOpts,
                         oneapi::mkl::transpose rOpts, int M, int N, int K,
                         const T *alpha, const arrayfire::oneapi::Array<T> &lhs,
                         dim_t lStride, const arrayfire::oneapi::Array<T> &rhs,
                         dim_t rStride, const T *beta,
                         arrayfire::oneapi::Array<T> &out, dim_t oleading) {
    using Dt                = arrayfire::oneapi::data_t<T>;
    const af::dim4 lStrides = lhs.strides();

    const af::dim4 rStrides    = rhs.strides();
    const af::dim4 oStrides    = out.strides();
    sycl::buffer<Dt, 1> lhsBuf = lhs.template getBufferWithOffset<Dt>();
    sycl::buffer<Dt, 1> rhsBuf = rhs.template getBufferWithOffset<Dt>();
    sycl::buffer<Dt, 1> outBuf = out.template getBufferWithOffset<Dt>();
    ::oneapi::mkl::blas::gemm(queue, lOpts, rOpts, M, N, K, *alpha, lhsBuf,
                              lStride, rhsBuf, rStride, *beta, outBuf,
                              oleading);
}

namespace arrayfire {
namespace oneapi {

void initBlas() { /*gpu_blas_init();*/
}

void deInitBlas() { /*gpu_blas_deinit();*/
}

bool isStrideMonotonic(const af::dim4 &dim) {
    return (dim[0] <= dim[1]) && (dim[1] <= dim[2]) && (dim[2] <= dim[3]);
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

    if (oDims.ndims() <= 2) {  // if non-batched
        if (rhs.dims()[bColDim] == 1) {
            if constexpr (std::is_same_v<T, arrayfire::common::half>) {
                // currently no half support for gemv, use gemm instead
                gemmDispatch<T>(getQueue(), lOpts, rOpts, M, N, K, alpha, lhs,
                                lStrides[1], rhs, rStrides[1], beta, out,
                                oStrides[1]);
            } else {
                dim_t incr =
                    (optRhs == AF_MAT_NONE) ? rStrides[0] : rStrides[1];
                gemvDispatch<T>(getQueue(), lOpts, rOpts, lDims[0], lDims[1],
                                alpha, lhs, lStrides[1], rhs, incr, beta, out,
                                oStrides[0]);
            }
        } else {
            gemmDispatch<T>(getQueue(), lOpts, rOpts, M, N, K, alpha, lhs,
                            lStrides[1], rhs, rStrides[1], beta, out,
                            oStrides[1]);
        }
    } else {  // if batched
        using Dt = arrayfire::oneapi::data_t<T>;

        int64_t batchSize = static_cast<int64_t>(oDims[2] * oDims[3]);

        bool is_l_d2_batched = (oDims[2] == lDims[2]) && lDims[2] != 1;
        bool is_l_d3_batched = (oDims[3] == lDims[3]) && lDims[3] != 1;
        bool is_r_d2_batched = (oDims[2] == rDims[2]) && rDims[2] != 1;
        bool is_r_d3_batched = (oDims[3] == rDims[3]) && rDims[3] != 1;

        // MKL requires stridec >= ldc * n, which may not be true with reordered
        // outputs if the stride is monotonic, then MKL requirements for
        // batching can be met
        bool canBatchMKL = isStrideMonotonic(oStrides);
        if (canBatchMKL) {
            sycl::buffer<Dt, 1> lhsBuf = lhs.template getBufferWithOffset<Dt>();
            sycl::buffer<Dt, 1> rhsBuf = rhs.template getBufferWithOffset<Dt>();
            sycl::buffer<Dt, 1> outBuf = out.template getBufferWithOffset<Dt>();

            const int64_t lda = lStrides[1];
            const int64_t ldb = rStrides[1];
            const int64_t ldc = oStrides[1];

            dim_t lstride = (is_l_d2_batched) ? lStrides[2]
                            : is_l_d3_batched ? lStrides[3]
                                              : 0;
            dim_t rstride = (is_r_d2_batched) ? rStrides[2]
                            : is_r_d3_batched ? rStrides[3]
                                              : 0;

            ::oneapi::mkl::blas::gemm_batch(getQueue(), lOpts, rOpts, M, N, K,
                                            *alpha, lhsBuf, lda, lstride,
                                            rhsBuf, ldb, rstride, *beta, outBuf,
                                            ldc, oStrides[2], batchSize);
        } else {
            std::vector<sycl::buffer<Dt>> lptrs;
            std::vector<sycl::buffer<Dt>> rptrs;
            std::vector<sycl::buffer<Dt>> optrs;

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

                lptrs.emplace_back(lhs.template getBufferWithOffset<Dt>(loff));
                rptrs.emplace_back(rhs.template getBufferWithOffset<Dt>(roff));
                optrs.emplace_back(out.template getBufferWithOffset<Dt>(zoff));
            }

            for (int n = 0; n < batchSize; n++) {
                ::oneapi::mkl::blas::gemm(getQueue(), lOpts, rOpts, M, N, K,
                                          *alpha, lptrs[n], lStrides[1],
                                          rptrs[n], rStrides[1], *beta,
                                          optrs[n], oStrides[1]);
            }
        }
    }
    ONEAPI_DEBUG_FINISH(getQueue());
}

template<typename T>
Array<T> dot(const Array<T> &lhs, const Array<T> &rhs, af_mat_prop optLhs,
             af_mat_prop optRhs) {
    auto lhs_ = (optLhs == AF_MAT_NONE ? lhs : conj<T>(lhs));
    auto rhs_ = (optRhs == AF_MAT_NONE ? rhs : conj<T>(rhs));
    auto temp = arithOp<T, af_mul_t>(lhs_, rhs_, lhs_.dims());
    return reduce<af_add_t, T, T>(temp, 0, false, 0);
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
INSTANTIATE_GEMM(half)

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
