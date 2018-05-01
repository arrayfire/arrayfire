/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/statistics.h>
#include "symbol_manager.hpp"

af_err af_mean(af_array *out, const af_array in, const dim_t dim)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, dim);
}

af_err af_mean_weighted(af_array *out, const af_array in, const af_array weights, const dim_t dim)
{
    CHECK_ARRAYS(in, weights);
    return CALL(out, in, weights, dim);
}

af_err af_var(af_array *out, const af_array in, const bool isbiased, const dim_t dim)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, isbiased, dim);
}

af_err af_var_weighted(af_array *out, const af_array in, const af_array weights, const dim_t dim)
{
    CHECK_ARRAYS(in, weights);
    return CALL(out, in, weights, dim);
}

af_err af_meanvar(af_array *mean, af_array *var, const af_array in,
                  const af_array weights, const af_var_bias bias, const dim_t dim)
{
    CHECK_ARRAYS(in, weights);
    return CALL(mean, var, in, weights, bias, dim);
}

af_err af_stdev(af_array *out, const af_array in, const dim_t dim)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, dim);
}

af_err af_cov(af_array* out, const af_array X, const af_array Y, const bool isbiased)
{
    CHECK_ARRAYS(X, Y);
    return CALL(out, X, Y, isbiased);
}

af_err af_median(af_array* out, const af_array in, const dim_t dim)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, dim);
}

af_err af_mean_all(double *real, double *imag, const af_array in)
{
    CHECK_ARRAYS(in);
    return CALL(real, imag, in);
}

af_err af_mean_all_weighted(double *real, double *imag, const af_array in, const af_array weights)
{
    CHECK_ARRAYS(in, weights);
    return CALL(real, imag, in, weights);
}

af_err af_var_all(double *realVal, double *imagVal, const af_array in, const bool isbiased)
{
    CHECK_ARRAYS(in);
    return CALL(realVal, imagVal, in, isbiased);
}

af_err af_var_all_weighted(double *realVal, double *imagVal, const af_array in, const af_array weights)
{
    CHECK_ARRAYS(in, weights);
    return CALL(realVal, imagVal, in, weights);
}

af_err af_stdev_all(double *real, double *imag, const af_array in)
{
    CHECK_ARRAYS(in);
    return CALL(real, imag, in);
}

af_err af_median_all(double *realVal, double *imagVal, const af_array in)
{
    CHECK_ARRAYS(in);
    return CALL(realVal, imagVal, in);
}

af_err af_corrcoef(double *realVal, double *imagVal, const af_array X, const af_array Y)
{
    CHECK_ARRAYS(X, Y);
    return CALL(realVal, imagVal, X, Y);
}

af_err af_topk(af_array *values, af_array *indices, const af_array in,
               const int k, const int dim, const af_topk_function order)
{
    CHECK_ARRAYS(in);
    return CALL(values, indices, in, k, dim, order);
}
