/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/lapack.h>
#include "symbol_manager.hpp"

af_err af_svd(af_array *u, af_array *s, af_array *vt, const af_array in) {
    CHECK_ARRAYS(in);
    CALL(af_svd, u, s, vt, in);
}

af_err af_svd_inplace(af_array *u, af_array *s, af_array *vt, af_array in) {
    CHECK_ARRAYS(in);
    CALL(af_svd_inplace, u, s, vt, in);
}

af_err af_lu(af_array *lower, af_array *upper, af_array *pivot,
             const af_array in) {
    CHECK_ARRAYS(in);
    CALL(af_lu, lower, upper, pivot, in);
}

af_err af_lu_inplace(af_array *pivot, af_array in, const bool is_lapack_piv) {
    CHECK_ARRAYS(in);
    CALL(af_lu_inplace, pivot, in, is_lapack_piv);
}

af_err af_qr(af_array *q, af_array *r, af_array *tau, const af_array in) {
    CHECK_ARRAYS(in);
    CALL(af_qr, q, r, tau, in);
}

af_err af_qr_inplace(af_array *tau, af_array in) {
    CHECK_ARRAYS(in);
    CALL(af_qr_inplace, tau, in);
}

af_err af_cholesky(af_array *out, int *info, const af_array in,
                   const bool is_upper) {
    CHECK_ARRAYS(in);
    CALL(af_cholesky, out, info, in, is_upper);
}

af_err af_cholesky_inplace(int *info, af_array in, const bool is_upper) {
    CHECK_ARRAYS(in);
    CALL(af_cholesky_inplace, info, in, is_upper);
}

af_err af_solve(af_array *x, const af_array a, const af_array b,
                const af_mat_prop options) {
    CHECK_ARRAYS(a, b);
    CALL(af_solve, x, a, b, options);
}

af_err af_solve_lu(af_array *x, const af_array a, const af_array piv,
                   const af_array b, const af_mat_prop options) {
    CHECK_ARRAYS(a, piv, b);
    CALL(af_solve_lu, x, a, piv, b, options);
}

af_err af_inverse(af_array *out, const af_array in, const af_mat_prop options) {
    CHECK_ARRAYS(in);
    CALL(af_inverse, out, in, options);
}

af_err af_pinverse(af_array *out, const af_array in, const double tol,
                   const af_mat_prop options) {
    CHECK_ARRAYS(in);
    CALL(af_pinverse, out, in, tol, options);
}

af_err af_rank(unsigned *rank, const af_array in, const double tol) {
    CHECK_ARRAYS(in);
    CALL(af_rank, rank, in, tol);
}

af_err af_det(double *det_real, double *det_imag, const af_array in) {
    CHECK_ARRAYS(in);
    CALL(af_det, det_real, det_imag, in);
}

af_err af_norm(double *out, const af_array in, const af_norm_type type,
               const double p, const double q) {
    CHECK_ARRAYS(in);
    CALL(af_norm, out, in, type, p, q);
}

af_err af_is_lapack_available(bool *out) { CALL(af_is_lapack_available, out); }
