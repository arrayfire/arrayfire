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
static void gemvDispatch(sycl::queue queue,
                         oneapi::mkl::transpose lOpts,
                         oneapi::mkl::transpose rOpts,
                         int M, int N, const T *alpha,
                         const arrayfire::oneapi::Array<T> &lhs, dim_t lStride,
                         const arrayfire::oneapi::Array<T> &x, dim_t incx,
                         const T *beta, arrayfire::oneapi::Array<T> &out,
                         dim_t oInc) {
    using Dt                   = arrayfire::oneapi::data_t<T>;
    const af::dim4 lStrides = lhs.strides();
    const af::dim4 xStrides = x.strides();
    const af::dim4 oStrides = out.strides();
    sycl::buffer<Dt, 1> lhsBuf = lhs.template getBufferWithOffset<Dt>();
    sycl::buffer<Dt, 1> xBuf   =   x.template getBufferWithOffset<Dt>();
    sycl::buffer<Dt, 1> outBuf = out.template getBufferWithOffset<Dt>();
    if constexpr(!std::is_same_v<T, arrayfire::common::half>) {
        ::oneapi::mkl::blas::gemv(queue, lOpts, (int64_t)M, (int64_t)N, (T)*alpha,
                                lhsBuf, (int64_t)lStride, xBuf, (int64_t)incx,
                                (T)*beta, outBuf, (int64_t)oInc);
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
    const af::dim4 rStrides = rhs.strides();
    const af::dim4 oStrides = out.strides();
    sycl::buffer<Dt, 1> lhsBuf = lhs.template getBufferWithOffset<Dt>();
    sycl::buffer<Dt, 1> rhsBuf = rhs.template getBufferWithOffset<Dt>();
    sycl::buffer<Dt, 1> outBuf = out.template getBufferWithOffset<Dt>();
    try {
    ::oneapi::mkl::blas::gemm(queue, lOpts, rOpts, M, N, K, *alpha, lhsBuf,
                              lStride, rhsBuf, rStride, *beta, outBuf,
                              oleading);
    queue.wait_and_throw();
    } catch(sycl::exception &e)  {
        std::cout << e.what() << std::endl;
    }
}

namespace arrayfire {
namespace oneapi {

void initBlas() { /*gpu_blas_init();*/
}

void deInitBlas() { /*gpu_blas_deinit();*/
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
    try{

    if (oDims.ndims() <= 2) {  // if non-batched
        if (rhs.dims()[bColDim] == 1) {
            if constexpr(!std::is_same_v<T, arrayfire::common::half>) {
                dim_t incr = (optRhs == AF_MAT_NONE) ? rStrides[0] : rStrides[1];
                gemvDispatch<T>(getQueue(), lOpts, rOpts, lDims[0], lDims[1], alpha, lhs,
                                lStrides[1], rhs, incr, beta, out, oStrides[0]);
            } else {
                gemmDispatch<T>(getQueue(), lOpts, rOpts, M, N, K, alpha, lhs,
                                lStrides[1], rhs, rStrides[1], beta, out,
                                oStrides[1]);
            }
        } else {
            printf("%d %d %d, l%d R%d o%d\n",  M, N, K, lStrides[1], rStrides[1], oStrides[1]);
            gemmDispatch<T>(getQueue(), lOpts, rOpts, M, N, K, alpha, lhs,
                            lStrides[1], rhs, rStrides[1], beta, out,
                            oStrides[1]);
        }
    } else {  // if batched
        using Dt = arrayfire::oneapi::data_t<T>;

        sycl::buffer<Dt, 1> lhsBuf = lhs.template getBufferWithOffset<Dt>();
        sycl::buffer<Dt, 1> rhsBuf = rhs.template getBufferWithOffset<Dt>();
        sycl::buffer<Dt, 1> outBuf = out.template getBufferWithOffset<Dt>();

        const int64_t lda = lStrides[1];
        const int64_t ldb = rStrides[1];
        const int64_t ldc = oStrides[1];

        int64_t batchSize = static_cast<int64_t>(oDims[2] * oDims[3]);

        bool is_l_d2_batched = (oDims[2] == lDims[2]) && lDims[2] != 1;
        bool is_l_d3_batched = (oDims[3] == lDims[3]) && lDims[3] != 1;
        bool is_r_d2_batched = (oDims[2] == rDims[2]) && rDims[2] != 1;
        bool is_r_d3_batched = (oDims[3] == rDims[3]) && rDims[3] != 1;

        std::cout << lStrides << std::endl;
        std::cout << rStrides << std::endl;

        const bool not_l_batched =
            (oDims[2] != lDims[2] && oDims[3] != lDims[3]);
        const bool not_r_batched =
            (oDims[2] != rDims[2] && oDims[3] != rDims[3]);

        //dim_t lstride = !not_l_batched ? 0 : (is_l_d2_batched) ? lStrides[2] : lStrides[3];
        //dim_t rstride = !not_r_batched ? 0 : (is_r_d2_batched) ? rStrides[2] : rStrides[3];
        dim_t lstride = (is_l_d2_batched) ? lStrides[2] : is_l_d3_batched ? lStrides[3] : 0;
        dim_t rstride = (is_r_d2_batched) ? rStrides[2] : is_r_d3_batched ? rStrides[3] : 0;

        ::oneapi::mkl::blas::gemm_batch(
            getQueue(), lOpts, rOpts, M, N, K, *alpha, lhsBuf, lda,
            lstride, rhsBuf, ldb,
            rstride, *beta, outBuf, ldc, oStrides[2],
            batchSize);
    }

        getQueue().wait_and_throw();
    } catch(sycl::exception &e)  {
        std::cout << e.what() << std::endl;
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
