/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <complex>

#include <Array.hpp>
#include <arith.hpp>
#include <blas.hpp>
#include <complex.hpp>
#include <err_opencl.hpp>
#include <math.hpp>
#include <reduce.hpp>
#include <transpose.hpp>

// Includes one of the supported OpenCL BLAS back-ends (e.g. clBLAS, CLBlast)
#include <magma/magma_blas.h>

#if defined(WITH_LINEAR_ALGEBRA)
#include <cpu/cpu_blas.hpp>
#endif

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
Array<T> matmul(const Array<T> &lhs, const Array<T> &rhs, af_mat_prop optLhs,
                af_mat_prop optRhs) {
#if defined(WITH_LINEAR_ALGEBRA)
    if (OpenCLCPUOffload(
            false)) {  // Do not force offload gemm on OSX Intel devices
        return cpu::matmul(lhs, rhs, optLhs, optRhs);
    }
#endif
    const auto lOpts = toBlasTranspose(optLhs);
    const auto rOpts = toBlasTranspose(optRhs);

    const auto aRowDim = (lOpts == OPENCL_BLAS_NO_TRANS) ? 0 : 1;
    const auto aColDim = (lOpts == OPENCL_BLAS_NO_TRANS) ? 1 : 0;
    const auto bColDim = (rOpts == OPENCL_BLAS_NO_TRANS) ? 1 : 0;

    const dim4 lDims = lhs.dims();
    const dim4 rDims = rhs.dims();
    const int M      = lDims[aRowDim];
    const int N      = rDims[bColDim];
    const int K      = lDims[aColDim];

    dim_t d2     = std::max(lDims[2], rDims[2]);
    dim_t d3     = std::max(lDims[3], rDims[3]);
    dim4 oDims   = af::dim4(M, N, d2, d3);
    Array<T> out = createEmptyArray<T>(oDims);

    const auto alpha = scalar<T>(1);
    const auto beta  = scalar<T>(0);

    const dim4 lStrides = lhs.strides();
    const dim4 rStrides = rhs.strides();
    const dim4 oStrides = out.strides();

    int batchSize = oDims[2] * oDims[3];

    bool is_l_d2_batched = oDims[2] == lDims[2];
    bool is_l_d3_batched = oDims[3] == lDims[3];
    bool is_r_d2_batched = oDims[2] == rDims[2];
    bool is_r_d3_batched = oDims[3] == rDims[3];

    for (int n = 0; n < batchSize; n++) {
        int w = n / oDims[2];
        int z = n - w * oDims[2];

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
            OPENCL_BLAS_CHECK(gemv(lOpts, lDims[0], lDims[1], alpha,
                                   (*lhs.get())(), lOffset, lStrides[1],
                                   (*rhs.get())(), rOffset, incr, beta,
                                   (*out.get())(), oOffset, 1, 1, &getQueue()(),
                                   0, nullptr, &event()));
        } else {
            gpu_blas_gemm_func<T> gemm;
            OPENCL_BLAS_CHECK(gemm(lOpts, rOpts, M, N, K, alpha, (*lhs.get())(),
                                   lOffset, lStrides[1], (*rhs.get())(),
                                   rOffset, rStrides[1], beta, (*out.get())(),
                                   oOffset, out.dims()[0], 1, &getQueue()(), 0,
                                   nullptr, &event()));
        }
    }

    return out;
}

template<typename T>
Array<T> dot(const Array<T> &lhs, const Array<T> &rhs, af_mat_prop optLhs,
             af_mat_prop optRhs) {
    const Array<T> lhs_ = (optLhs == AF_MAT_NONE ? lhs : conj<T>(lhs));
    const Array<T> rhs_ = (optRhs == AF_MAT_NONE ? rhs : conj<T>(rhs));

    const Array<T> temp = arithOp<T, af_mul_t>(lhs_, rhs_, lhs_.dims());
    return reduce<af_add_t, T, T>(temp, 0, false, 0);
}

#define INSTANTIATE_BLAS(TYPE)                                \
    template Array<TYPE> matmul<TYPE>(const Array<TYPE> &lhs, \
                                      const Array<TYPE> &rhs, \
                                      af_mat_prop optLhs, af_mat_prop optRhs);

INSTANTIATE_BLAS(float)
INSTANTIATE_BLAS(cfloat)
INSTANTIATE_BLAS(double)
INSTANTIATE_BLAS(cdouble)

#define INSTANTIATE_DOT(TYPE)                                                  \
    template Array<TYPE> dot<TYPE>(const Array<TYPE> &lhs,                     \
                                   const Array<TYPE> &rhs, af_mat_prop optLhs, \
                                   af_mat_prop optRhs);

INSTANTIATE_DOT(float)
INSTANTIATE_DOT(double)
INSTANTIATE_DOT(cfloat)
INSTANTIATE_DOT(cdouble)

}  // namespace opencl
