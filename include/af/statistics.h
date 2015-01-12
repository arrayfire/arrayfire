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

AFAPI array mean(const array& in, dim_type dim=0);

template<typename T>
AFAPI T mean(const array& in);

AFAPI array var(const array& in, bool isbiased=false, int dim=-1);

AFAPI array var(const array& in, const array weights, int dim=-1);

template<typename T>
AFAPI T var(const array& in, bool isbiased=false);

template<typename T>
AFAPI T var(const array& in, const array weights);

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

AFAPI af_err af_mean(af_array *out, const af_array in, dim_type dim);

AFAPI af_err af_mean_all(double *real, double *imag, const af_array in);

AFAPI af_err af_var(af_array *out, const af_array& in, bool isbiased, int dim);

AFAPI af_err af_var_weighted(af_array *out, const af_array& in,
    const af_array weights);

AFAPI af_err af_var_all(double *realVal, double *imagVal, const af_array in,
    bool isbiased);

AFAPI af_err af_var_all_weighted(double *realVal, double *imagVal,
    const af_array in, const af_array weights);

#ifdef __cplusplus
}
#endif
