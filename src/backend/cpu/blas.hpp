/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <af/defines.h>

namespace arrayfire {
namespace cpu {

template<typename T>
void gemm(Array<T> &out, af_mat_prop optLhs, af_mat_prop optRhs, const T *alpha,
          const Array<T> &lhs, const Array<T> &rhs, const T *beta);

template<typename T>
Array<T> matmul(const Array<T> &lhs, const Array<T> &rhs, af_mat_prop optLhs,
                af_mat_prop optRhs) {
    int Mdim     = optLhs == AF_MAT_NONE ? 0 : 1;
    int Ndim     = optRhs == AF_MAT_NONE ? 1 : 0;
    Array<T> res = createEmptyArray<T>(
        dim4(lhs.dims()[Mdim], rhs.dims()[Ndim], lhs.dims()[2], lhs.dims()[3]));
    static const T alpha = T(1.0);
    static const T beta  = T(0.0);
    gemm(res, optLhs, optRhs, &alpha, lhs, rhs, &beta);
    return res;
}

template<typename T>
Array<T> dot(const Array<T> &lhs, const Array<T> &rhs, af_mat_prop optLhs,
             af_mat_prop optRhs);

}  // namespace cpu
}  // namespace arrayfire
