/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/deprecated.hpp>
#include <af/array.h>
#include <af/statistics.h>
#include "symbol_manager.hpp"

af_err af_mean(af_array *out, const af_array in, const dim_t dim) {
    CHECK_ARRAYS(in);
    CALL(af_mean, out, in, dim);
}

af_err af_mean_weighted(af_array *out, const af_array in,
                        const af_array weights, const dim_t dim) {
    CHECK_ARRAYS(in, weights);
    CALL(af_mean_weighted, out, in, weights, dim);
}

AF_DEPRECATED_WARNINGS_OFF
af_err af_var(af_array *out, const af_array in, const bool isbiased,
              const dim_t dim) {
    CHECK_ARRAYS(in);
    CALL(af_var, out, in, isbiased, dim);
}
AF_DEPRECATED_WARNINGS_ON

af_err af_var_weighted(af_array *out, const af_array in, const af_array weights,
                       const dim_t dim) {
    CHECK_ARRAYS(in, weights);
    CALL(af_var_weighted, out, in, weights, dim);
}

af_err af_meanvar(af_array *mean, af_array *var, const af_array in,
                  const af_array weights, const af_var_bias bias,
                  const dim_t dim) {
    CHECK_ARRAYS(in, weights);
    CALL(af_meanvar, mean, var, in, weights, bias, dim);
}

AF_DEPRECATED_WARNINGS_OFF
af_err af_stdev(af_array *out, const af_array in, const dim_t dim) {
    CHECK_ARRAYS(in);
    CALL(af_stdev, out, in, dim);
}

af_err af_cov(af_array *out, const af_array X, const af_array Y,
              const bool isbiased) {
    CHECK_ARRAYS(X, Y);
    CALL(af_cov, out, X, Y, isbiased);
}
AF_DEPRECATED_WARNINGS_ON

af_err af_median(af_array *out, const af_array in, const dim_t dim) {
    CHECK_ARRAYS(in);
    CALL(af_median, out, in, dim);
}

af_err af_mean_all(double *real, double *imag, const af_array in) {
    CHECK_ARRAYS(in);
    CALL(af_mean_all, real, imag, in);
}

af_err af_mean_all_weighted(double *real, double *imag, const af_array in,
                            const af_array weights) {
    CHECK_ARRAYS(in, weights);
    CALL(af_mean_all_weighted, real, imag, in, weights);
}

AF_DEPRECATED_WARNINGS_OFF
af_err af_var_all(double *realVal, double *imagVal, const af_array in,
                  const bool isbiased) {
    CHECK_ARRAYS(in);
    CALL(af_var_all, realVal, imagVal, in, isbiased);
}
AF_DEPRECATED_WARNINGS_ON

af_err af_var_all_weighted(double *realVal, double *imagVal, const af_array in,
                           const af_array weights) {
    CHECK_ARRAYS(in, weights);
    CALL(af_var_all_weighted, realVal, imagVal, in, weights);
}

AF_DEPRECATED_WARNINGS_OFF
af_err af_stdev_all(double *real, double *imag, const af_array in) {
    CHECK_ARRAYS(in);
    CALL(af_stdev_all, real, imag, in);
}
AF_DEPRECATED_WARNINGS_ON

af_err af_median_all(double *realVal, double *imagVal, const af_array in) {
    CHECK_ARRAYS(in);
    CALL(af_median_all, realVal, imagVal, in);
}

af_err af_corrcoef(double *realVal, double *imagVal, const af_array X,
                   const af_array Y) {
    CHECK_ARRAYS(X, Y);
    CALL(af_corrcoef, realVal, imagVal, X, Y);
}

af_err af_topk(af_array *values, af_array *indices, const af_array in,
               const int k, const int dim, const af_topk_function order) {
    CHECK_ARRAYS(in);
    CALL(af_topk, values, indices, in, k, dim, order);
}

af_err af_var_v2(af_array *out, const af_array in, const af_var_bias bias,
                 const dim_t dim) {
    CHECK_ARRAYS(in);
    CALL(af_var_v2, out, in, bias, dim);
}

af_err af_var_all_v2(double *realVal, double *imagVal, const af_array in,
                     const af_var_bias bias) {
    CHECK_ARRAYS(in);
    CALL(af_var_all_v2, realVal, imagVal, in, bias);
}

af_err af_cov_v2(af_array *out, const af_array X, const af_array Y,
                 const af_var_bias bias) {
    CHECK_ARRAYS(X, Y);
    CALL(af_cov_v2, out, X, Y, bias);
}

af_err af_stdev_v2(af_array *out, const af_array in, const af_var_bias bias,
                   const dim_t dim) {
    CHECK_ARRAYS(in);
    CALL(af_stdev_v2, out, in, bias, dim);
}

af_err af_stdev_all_v2(double *real, double *imag, const af_array in,
                       const af_var_bias bias) {
    CHECK_ARRAYS(in);
    CALL(af_stdev_all_v2, real, imag, in, bias);
}
