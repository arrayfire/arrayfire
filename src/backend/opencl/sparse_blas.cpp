/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <sparse_blas.hpp>

#include <kernel/cscmm.hpp>
#include <kernel/cscmv.hpp>
#include <kernel/csrmm.hpp>
#include <kernel/csrmv.hpp>

#include <cassert>
#include <stdexcept>
#include <string>

#include <common/err_common.hpp>
#include <complex.hpp>
#include <err_opencl.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <transpose.hpp>
#include <af/dim4.hpp>

#if defined(WITH_LINEAR_ALGEBRA)
#include <cpu/cpu_sparse_blas.hpp>
#endif  // WITH_LINEAR_ALGEBRA

namespace arrayfire {
namespace opencl {

using namespace common;

template<typename T>
Array<T> matmul(const common::SparseArray<T>& lhs, const Array<T>& rhsIn,
                af_mat_prop optLhs, af_mat_prop optRhs) {
#if defined(WITH_LINEAR_ALGEBRA)
    if (OpenCLCPUOffload(
            false)) {  // Do not force offload gemm on OSX Intel devices
        return cpu::matmul(lhs, rhsIn, optLhs, optRhs);
    }
#endif

    int lRowDim = (optLhs == AF_MAT_NONE) ? 0 : 1;
    // int lColDim = (optLhs == AF_MAT_NONE) ? 1 : 0;
    static const int rColDim =
        1;  // Unsupported : (optRhs == AF_MAT_NONE) ? 1 : 0;

    dim4 lDims = lhs.dims();
    dim4 rDims = rhsIn.dims();
    int M      = lDims[lRowDim];
    int N      = rDims[rColDim];
    // int K = lDims[lColDim];

    const Array<T> rhs =
        (N != 1 && optLhs == AF_MAT_NONE) ? transpose(rhsIn, false) : rhsIn;
    Array<T> out = createEmptyArray<T>(af::dim4(M, N, 1, 1));

    static const T alpha = scalar<T>(1.0);
    static const T beta  = scalar<T>(0.0);

    const Array<T>& values   = lhs.getValues();
    const Array<int>& rowIdx = lhs.getRowIdx();
    const Array<int>& colIdx = lhs.getColIdx();

    if (optLhs == AF_MAT_NONE) {
        if (N == 1) {
            kernel::csrmv(out, values, rowIdx, colIdx, rhs, alpha, beta);
        } else {
            kernel::csrmm_nt(out, values, rowIdx, colIdx, rhs, alpha, beta);
        }
    } else {
        // CSR transpose is a CSC matrix
        if (N == 1) {
            kernel::cscmv(out, values, rowIdx, colIdx, rhs, alpha, beta,
                          optLhs == AF_MAT_CTRANS);
        } else {
            kernel::cscmm_nn(out, values, rowIdx, colIdx, rhs, alpha, beta,
                             optLhs == AF_MAT_CTRANS);
        }
    }
    return out;
}

#define INSTANTIATE_SPARSE(T)                                            \
    template Array<T> matmul<T>(const common::SparseArray<T>& lhs,       \
                                const Array<T>& rhs, af_mat_prop optLhs, \
                                af_mat_prop optRhs);

INSTANTIATE_SPARSE(float)
INSTANTIATE_SPARSE(double)
INSTANTIATE_SPARSE(cfloat)
INSTANTIATE_SPARSE(cdouble)

}  // namespace opencl
}  // namespace arrayfire
