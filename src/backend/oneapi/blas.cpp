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
#include <common/traits.hpp>
#include <complex.hpp>
#include <err_oneapi.hpp>
#include <math.hpp>
#include <reduce.hpp>
#include <transpose.hpp>

#include <complex>
#include <vector>

using common::half;

namespace oneapi {

void initBlas() { /*gpu_blas_init();*/ }

void deInitBlas() { /*gpu_blas_deinit();*/ }

template<typename T>
void gemm_fallback(Array<T> &out, af_mat_prop optLhs, af_mat_prop optRhs,
                   const T *alpha, const Array<T> &lhs, const Array<T> &rhs,
                   const T *beta) {
    ONEAPI_NOT_SUPPORTED("");
}

template<>
void gemm_fallback<half>(Array<half> & /*out*/, af_mat_prop /*optLhs*/,
                         af_mat_prop /*optRhs*/, const half * /*alpha*/,
                         const Array<half> & /*lhs*/,
                         const Array<half> & /*rhs*/, const half * /*beta*/) {
    ONEAPI_NOT_SUPPORTED("");
    assert(false && "CPU fallback not implemented for f16");
}

template<typename T>
void gemm(Array<T> &out, af_mat_prop optLhs, af_mat_prop optRhs, const T *alpha,
          const Array<T> &lhs, const Array<T> &rhs, const T *beta) {
    ONEAPI_NOT_SUPPORTED("");
}

template<typename T>
Array<T> dot(const Array<T> &lhs, const Array<T> &rhs, af_mat_prop optLhs,
             af_mat_prop optRhs) {
    ONEAPI_NOT_SUPPORTED("");
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
