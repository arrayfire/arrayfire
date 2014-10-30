/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>
#include <af/array.h>

#define AF_MAX_DIMS 4

#ifdef __cplusplus
namespace af
{
    // Add all the elements along a dimension
    AFAPI array sum(const array &in, const int dim = 0);

    // Get the minimum of all elements along a dimension
    AFAPI array min(const array &in, const int dim = 0);

    // Get the maximum of all elements along a dimension
    AFAPI array max(const array &in, const int dim = 0);

    // Check if all elements along a dimension are true
    AFAPI array alltrue(const array &in, const int dim = 0);

    // Check if any elements along a dimension are true
    AFAPI array anytrue(const array &in, const int dim = 0);

    // Count number of non zero elements along a dimension
    AFAPI array count(const array &in, const int dim = 0);
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
    // Add all the elements along a dimension
    AFAPI af_err af_sum(af_array *out, const af_array in, const int dim);

    // Get the minimum of all elements along a dimension
    AFAPI af_err af_min(af_array *out, const af_array in, const int dim);

    // Get the maximum of all elements along a dimension
    AFAPI af_err af_max(af_array *out, const af_array in, const int dim);

    // Check if all elements along a dimension are true
    AFAPI af_err af_alltrue(af_array *out, const af_array in, const int dim);

    // Check if any elements along a dimension are true
    AFAPI af_err af_anytrue(af_array *out, const af_array in, const int dim);

    // Count number of non zero elements along a dimension
    AFAPI af_err af_count(af_array *out, const af_array in, const int dim);

#ifdef __cplusplus
}
#endif
