/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include "array.h"

#ifdef __cplusplus
namespace af
{

AFAPI array mean(const array& in, dim_type dim=-1);

AFAPI array mean(const array& in, const array& weights, dim_type dim=-1);

template<typename T>
AFAPI T mean(const array& in);

template<typename T>
AFAPI T mean(const array& in, const array& weights);


template<typename T>
AFAPI T stdev(const array& in);

AFAPI array stdev(const array& in, dim_type dim=-1);


AFAPI array var(const array& in, bool isbiased=false, dim_type dim=-1);

AFAPI array var(const array& in, const array weights, dim_type dim=-1);

template<typename T>
AFAPI T var(const array& in, bool isbiased=false);

template<typename T>
AFAPI T var(const array& in, const array weights);


AFAPI array cov(const array& X, const array& Y, bool isbiased=false);


template<typename T>
AFAPI T median(const array& in);

AFAPI array median(const array& in, dim_type dim=-1);


template<typename T>
AFAPI T corrcoef(const array& X, const array& Y);

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

AFAPI af_err af_mean(af_array *out, const af_array in, dim_type dim);

AFAPI af_err af_mean_weighted(af_array *out, const af_array in, const af_array weights, dim_type dim);

AFAPI af_err af_mean_all(double *real, double *imag, const af_array in);

AFAPI af_err af_mean_all_weighted(double *real, double *imag, const af_array in, const af_array weights);


AFAPI af_err af_stdev_all(double *real, double *imag, const af_array in);

AFAPI af_err af_stdev(af_array *out, const af_array in, dim_type dim);


AFAPI af_err af_var(af_array *out, const af_array in, bool isbiased, dim_type dim);

AFAPI af_err af_var_weighted(af_array *out, const af_array in, const af_array weights, dim_type dim);

AFAPI af_err af_var_all(double *realVal, double *imagVal, const af_array in, bool isbiased);

AFAPI af_err af_var_all_weighted(double *realVal, double *imagVal, const af_array in, const af_array weights);


AFAPI af_err af_cov(af_array* out, const af_array X, const af_array Y, bool isbiased);


AFAPI af_err af_median_all(double *realVal, double *imagVal, const af_array in);

AFAPI af_err af_median(af_array* out, const af_array in, dim_type dim);


AFAPI af_err af_corrcoef(double *realVal, double *imagVal, const af_array X, const af_array Y);

#ifdef __cplusplus
}
#endif
