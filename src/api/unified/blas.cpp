/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/blas.h>
#include "symbol_manager.hpp"

AFAPI af_err af_gemm(af_array *out, const af_mat_prop optLhs,
                     const af_mat_prop optRhs, const void *alpha,
                     const af_array lhs, const af_array rhs, const void *beta) {
    CHECK_ARRAYS(out, lhs, rhs);
    CALL(af_gemm, out, optLhs, optRhs, alpha, lhs, rhs, beta);
}

af_err af_matmul(af_array *out, const af_array lhs, const af_array rhs,
                 const af_mat_prop optLhs, const af_mat_prop optRhs) {
    CHECK_ARRAYS(lhs, rhs);
    CALL(af_matmul, out, lhs, rhs, optLhs, optRhs);
}

af_err af_dot(af_array *out, const af_array lhs, const af_array rhs,
              const af_mat_prop optLhs, const af_mat_prop optRhs) {
    CHECK_ARRAYS(lhs, rhs);
    CALL(af_dot, out, lhs, rhs, optLhs, optRhs);
}

af_err af_dot_all(double *rval, double *ival, const af_array lhs,
                  const af_array rhs, const af_mat_prop optLhs,
                  const af_mat_prop optRhs) {
    CHECK_ARRAYS(lhs, rhs);
    CALL(af_dot_all, rval, ival, lhs, rhs, optLhs, optRhs);
}

af_err af_transpose(af_array *out, af_array in, const bool conjugate) {
    CHECK_ARRAYS(in);
    CALL(af_transpose, out, in, conjugate);
}

af_err af_transpose_inplace(af_array in, const bool conjugate) {
    CHECK_ARRAYS(in);
    CALL(af_transpose_inplace, in, conjugate);
}
