/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <complex>

#include <blas.hpp>
#include <Array.hpp>
#include <err_opencl.hpp>
#include <math.hpp>
#include <transpose.hpp>
#include <arith.hpp>
#include <reduce.hpp>
#include <complex.hpp>

// Includes one of the supported OpenCL BLAS back-ends (e.g. clBLAS, CLBlast)
#include <magma/magma_blas.h>

#if defined(WITH_OPENCL_LINEAR_ALGEBRA)
#include <cpu/cpu_blas.hpp>
#endif

namespace opencl
{

// Converts an af_mat_prop options to a transpose type for one of the OpenCL BLAS back-ends
OPENCL_BLAS_TRANS_T
toBlasTranspose(af_mat_prop opt)
{
    switch(opt) {
        case AF_MAT_NONE    : return OPENCL_BLAS_NO_TRANS;
        case AF_MAT_TRANS   : return OPENCL_BLAS_TRANS;
        case AF_MAT_CTRANS  : return OPENCL_BLAS_CONJ_TRANS;
        default             : AF_ERROR("INVALID af_mat_prop", AF_ERR_ARG);
    }
}

// Initialization of the OpenCL BLAS library
void
initBlas()
{
    gpu_blas_init();
}

template<typename T>
Array<T> matmul(const Array<T> &lhs, const Array<T> &rhs,
                af_mat_prop optLhs, af_mat_prop optRhs)
{
#if defined(WITH_OPENCL_LINEAR_ALGEBRA)
    if(OpenCLCPUOffload(false)) {   // Do not force offload gemm on OSX Intel devices
        return cpu::matmul(lhs, rhs, optLhs, optRhs);
    }
#endif
    initBlas();

    const auto lOpts = toBlasTranspose(optLhs);
    const auto rOpts = toBlasTranspose(optRhs);

    const auto aRowDim = (lOpts == OPENCL_BLAS_NO_TRANS) ? 0 : 1;
    const auto aColDim = (lOpts == OPENCL_BLAS_NO_TRANS) ? 1 : 0;
    const auto bColDim = (rOpts == OPENCL_BLAS_NO_TRANS) ? 1 : 0;

    const dim4 lDims = lhs.dims();
    const dim4 rDims = rhs.dims();
    const int M = lDims[aRowDim];
    const int N = rDims[bColDim];
    const int K = lDims[aColDim];

    Array<T> out = createEmptyArray<T>(af::dim4(M, N, 1, 1));
    const auto alpha = scalar<T>(1);
    const auto beta  = scalar<T>(0);

    const dim4 lStrides = lhs.strides();
    const dim4 rStrides = rhs.strides();
    cl::Event event;
    if(rDims[bColDim] == 1) {
        gpu_blas_gemv_func<T> gemv;
        OPENCL_BLAS_CHECK(
            gemv(lOpts, lDims[0], lDims[1],
                 alpha,
                 (*lhs.get())(), lhs.getOffset(), lStrides[1],
                 (*rhs.get())(), rhs.getOffset(), rStrides[0],
                 beta,
                 (*out.get())(), out.getOffset(), 1,
                 1, &getQueue()(), 0, nullptr, &event())
        );
    } else {
        gpu_blas_gemm_func<T> gemm;
        OPENCL_BLAS_CHECK(
            gemm(lOpts, rOpts, M, N, K,
                 alpha,
                 (*lhs.get())(), lhs.getOffset(), lStrides[1],
                 (*rhs.get())(), rhs.getOffset(), rStrides[1],
                 beta,
                 (*out.get())(), out.getOffset(), out.dims()[0],
                 1, &getQueue()(), 0, nullptr, &event())
        );
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

#define INSTANTIATE_BLAS(TYPE)                                                          \
    template Array<TYPE> matmul<TYPE>(const Array<TYPE> &lhs, const Array<TYPE> &rhs,   \
                    af_mat_prop optLhs, af_mat_prop optRhs);

INSTANTIATE_BLAS(float)
INSTANTIATE_BLAS(cfloat)
INSTANTIATE_BLAS(double)
INSTANTIATE_BLAS(cdouble)

#define INSTANTIATE_DOT(TYPE)                                                       \
    template Array<TYPE> dot<TYPE>(const Array<TYPE> &lhs, const Array<TYPE> &rhs,  \
                                   af_mat_prop optLhs, af_mat_prop optRhs);

INSTANTIATE_DOT(float)
INSTANTIATE_DOT(double)
INSTANTIATE_DOT(cfloat)
INSTANTIATE_DOT(cdouble)

}
