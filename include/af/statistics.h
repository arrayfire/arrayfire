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

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

AFAPI af_err af_mean(af_array *out, const af_array in, dim_type dim);

AFAPI af_err af_mean_all(double *real, double *imag, const af_array in);

#ifdef __cplusplus
}
#endif
