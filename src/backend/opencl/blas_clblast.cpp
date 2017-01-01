/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined(USE_CLBLAST)

#include <complex>

#include <blas.hpp>
#include <Array.hpp>
#include <err_common.hpp>
#include <math.hpp>
#include <transpose.hpp>
#include <arith.hpp>
#include <reduce.hpp>
#include <complex.hpp>

#include <err_clblast.hpp>

#if defined(WITH_OPENCL_LINEAR_ALGEBRA)
#include <cpu/cpu_blas.hpp>
#endif

namespace opencl
{

void
initBlas()
{
  // Nothing to do here for CLBlast
}

clblast::Transpose
toClblastTranspose(af_mat_prop opt)
{
    switch(opt) {
        case AF_MAT_NONE    : return clblast::Transpose::kNo;
        case AF_MAT_TRANS   : return clblast::Transpose::kYes;
        case AF_MAT_CTRANS  : return clblast::Transpose::kConjugate;
        default             : AF_ERROR("INVALID af_mat_prop", AF_ERR_ARG);
    }
}

// Defines type conversions from ArrayFire (OpenCL) to CLBlast (C++ std)
template <typename T> struct CLBlastConstant { using Type = T; };
template <> struct CLBlastConstant<cfloat> { using Type = std::complex<float>; };
template <> struct CLBlastConstant<cdouble> { using Type = std::complex<double>; };

// Converts a constant from ArrayFire types (OpenCL) to CLBlast types (C++ std)
template <typename T> typename CLBlastConstant<T>::Type toCLBlastConstant(const T val);

// Specializations of the above function
template <> float toCLBlastConstant(const float val) { return val; }
template <> double toCLBlastConstant(const double val) { return val; }
template <> std::complex<float> toCLBlastConstant(cfloat val) { return {val.s[0], val.s[1]}; }
template <> std::complex<double> toCLBlastConstant(cdouble val) { return {val.s[0], val.s[1]}; }

template<typename T>
Array<T> matmul(const Array<T> &lhs, const Array<T> &rhs,
                af_mat_prop optLhs, af_mat_prop optRhs)
{
#if defined(WITH_OPENCL_LINEAR_ALGEBRA)
    if(OpenCLCPUOffload(false)) {   // Do not force offload gemm on OSX Intel devices
        return cpu::matmul(lhs, rhs, optLhs, optRhs);
    }
#endif

    const auto lOpts = toClblastTranspose(optLhs);
    const auto rOpts = toClblastTranspose(optRhs);

    const auto aRowDim = (lOpts == clblast::Transpose::kNo) ? 0 : 1;
    const auto aColDim = (lOpts == clblast::Transpose::kNo) ? 1 : 0;
    const auto bColDim = (rOpts == clblast::Transpose::kNo) ? 1 : 0;

    const dim4 lDims = lhs.dims();
    const dim4 rDims = rhs.dims();
    const int M = lDims[aRowDim];
    const int N = rDims[bColDim];
    const int K = lDims[aColDim];

    Array<T> out = createEmptyArray<T>(af::dim4(M, N, 1, 1));
    const auto alpha = scalar<T>(1);
    const auto beta  = scalar<T>(0);
    const auto alpha_clblast = toCLBlastConstant(alpha);
    const auto beta_clblast = toCLBlastConstant(beta);

    const dim4 lStrides = lhs.strides();
    const dim4 rStrides = rhs.strides();
    if(rDims[bColDim] == 1) {
        CLBLAST_CHECK(
            clblast::Gemv(clblast::Layout::kColMajor, lOpts,
                          lDims[0], lDims[1],
                          alpha_clblast,
                          (*lhs.get())(), lhs.getOffset(), lStrides[1],
                          (*rhs.get())(), rhs.getOffset(), rStrides[0],
                          beta_clblast,
                          (*out.get())(), out.getOffset(), 1,
                          &getQueue()())
        );
    } else {
        CLBLAST_CHECK(
            clblast::Gemm(clblast::Layout::kColMajor, lOpts, rOpts,
                          M, N, K,
                          alpha_clblast,
                          (*lhs.get())(), lhs.getOffset(), lStrides[1],
                          (*rhs.get())(), rhs.getOffset(), rStrides[1],
                          beta_clblast,
                          (*out.get())(), out.getOffset(), out.dims()[0],
                          &getQueue()())
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

#endif // USE_CLBLAST
