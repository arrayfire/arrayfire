/*******************************************************
 * Copyright (c) 2014, ArrayFire
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
#include <common/traits.hpp>
#include <complex.hpp>
#include <err_opencl.hpp>
#include <math.hpp>
#include <reduce.hpp>
#include <transpose.hpp>

#include <complex>
#include <vector>

// Includes one of the supported OpenCL BLAS back-ends (e.g. clBLAS, CLBlast)
#include <cpu/cpu_blas.hpp>
#include <magma/magma_blas.h>

using arrayfire::common::half;

namespace arrayfire {
namespace opencl {

void initBlas() { gpu_blas_init(); }

void deInitBlas() { gpu_blas_deinit(); }

// Converts an af_mat_prop options to a transpose type for one of the OpenCL
// BLAS back-ends
OPENCL_BLAS_TRANS_T
toBlasTranspose(af_mat_prop opt) {
    switch (opt) {
        case AF_MAT_NONE: return OPENCL_BLAS_NO_TRANS;
        case AF_MAT_TRANS: return OPENCL_BLAS_TRANS;
        case AF_MAT_CTRANS: return OPENCL_BLAS_CONJ_TRANS;
        default: AF_ERROR("INVALID af_mat_prop", AF_ERR_ARG);
    }
}

template<typename T>
void gemm_fallback(Array<T> &out, af_mat_prop optLhs, af_mat_prop optRhs,
                   const T *alpha, const Array<T> &lhs, const Array<T> &rhs,
                   const T *beta) {
    cpu::gemm(out, optLhs, optRhs, alpha, lhs, rhs, beta);
}

template<>
void gemm_fallback<half>(Array<half> & /*out*/, af_mat_prop /*optLhs*/,
                         af_mat_prop /*optRhs*/, const half * /*alpha*/,
                         const Array<half> & /*lhs*/,
                         const Array<half> & /*rhs*/, const half * /*beta*/) {
    assert(false && "CPU fallback not implemented for f16");
}

template<typename T>
void gemm(Array<T> &out, af_mat_prop optLhs, af_mat_prop optRhs, const T *alpha,
          const Array<T> &lhs, const Array<T> &rhs, const T *beta) {
#if defined(WITH_LINEAR_ALGEBRA)
    // Do not force offload gemm on OSX Intel devices
    if (OpenCLCPUOffload(false) &&
        static_cast<af_dtype>(dtype_traits<T>::af_type) != f16) {
        gemm_fallback(out, optLhs, optRhs, alpha, lhs, rhs, beta);
        return;
    }
#endif
    const auto lOpts = toBlasTranspose(optLhs);
    const auto rOpts = toBlasTranspose(optRhs);

    const auto aRowDim = (lOpts == OPENCL_BLAS_NO_TRANS) ? 0 : 1;
    const auto aColDim = (lOpts == OPENCL_BLAS_NO_TRANS) ? 1 : 0;
    const auto bColDim = (rOpts == OPENCL_BLAS_NO_TRANS) ? 1 : 0;

    const dim4 &lDims = lhs.dims();
    const dim4 &rDims = rhs.dims();
    const int M       = lDims[aRowDim];
    const int N       = rDims[bColDim];
    const int K       = lDims[aColDim];
    const dim4 oDims  = out.dims();

    const dim4 &lStrides = lhs.strides();
    const dim4 &rStrides = rhs.strides();
    const dim4 oStrides  = out.strides();

    int batchSize = static_cast<int>(oDims[2] * oDims[3]);

    bool is_l_d2_batched = oDims[2] == lDims[2];
    bool is_l_d3_batched = oDims[3] == lDims[3];
    bool is_r_d2_batched = oDims[2] == rDims[2];
    bool is_r_d3_batched = oDims[3] == rDims[3];

    for (int n = 0; n < batchSize; n++) {
        int w = static_cast<int>(n / oDims[2]);
        int z = static_cast<int>(n - w * oDims[2]);

        int loff = z * (is_l_d2_batched * lStrides[2]) +
                   w * (is_l_d3_batched * lStrides[3]);
        int roff = z * (is_r_d2_batched * rStrides[2]) +
                   w * (is_r_d3_batched * rStrides[3]);

        dim_t lOffset = lhs.getOffset() + loff;
        dim_t rOffset = rhs.getOffset() + roff;
        dim_t oOffset = out.getOffset() + z * oStrides[2] + w * oStrides[3];

        cl::Event event;
        if (rDims[bColDim] == 1) {
            dim_t incr = (optRhs == AF_MAT_NONE) ? rStrides[0] : rStrides[1];
            gpu_blas_gemv_func<T> gemv;
            OPENCL_BLAS_CHECK(gemv(lOpts, lDims[0], lDims[1], *alpha,
                                   (*lhs.get())(), lOffset, lStrides[1],
                                   (*rhs.get())(), rOffset, incr, *beta,
                                   (*out.get())(), oOffset, oStrides[0], 1,
                                   &getQueue()(), 0, nullptr, &event()));
        } else {
            gpu_blas_gemm_func<T> gemm;
            OPENCL_BLAS_CHECK(gemm(lOpts, rOpts, M, N, K, *alpha,
                                   (*lhs.get())(), lOffset, lStrides[1],
                                   (*rhs.get())(), rOffset, rStrides[1], *beta,
                                   (*out.get())(), oOffset, oStrides[1], 1,
                                   &getQueue()(), 0, nullptr, &event()));
        }
    }
}

template<typename T>
Array<T> dot(const Array<T> &lhs, const Array<T> &rhs, af_mat_prop optLhs,
             af_mat_prop optRhs) {
    const Array<T> lhs_ = (optLhs == AF_MAT_NONE ? lhs : conj<T>(lhs));
    const Array<T> rhs_ = (optRhs == AF_MAT_NONE ? rhs : conj<T>(rhs));

    const Array<T> temp = arithOp<T, af_mul_t>(lhs_, rhs_, lhs_.dims());
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

}  // namespace opencl
}  // namespace arrayfire
