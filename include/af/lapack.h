/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/array.h>
#include <af/defines.h>

#if __cplusplus
namespace af
{

    /**
       \defgroup manip_func_lin_algebra lin_algebra
       @{
    */
    AFAPI void lu(array &out, array &pivot, const array &in);

    AFAPI void lu(array &lower, array &upper, array &pivot, const array &in);

    AFAPI array luInPlace(array &in);

    AFAPI void qr(array &out, array &tau, const array &in);

    AFAPI void qr(array &q, array &r, array &tau, const array &in);

    AFAPI array qrInPlace(array &in);

    AFAPI array cholesky(const array &in, int *info = NULL, const bool is_upper = true);

    AFAPI void choleskyInPlace(array &in, int *info = NULL, const bool is_upper = true);

    AFAPI array solve(const array &a, const array &b, const af_solve_t options = AF_SOLVE_NONE);

    AFAPI array inverse(const array &in);

    /**
       @}
    */
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /**
       \ingroup manip_func_lin_algebra
    */
    AFAPI af_err af_lu(af_array *lower, af_array *upper, af_array *pivot, const af_array in);

    AFAPI af_err af_lu_inplace(af_array *pivot, af_array in);

    AFAPI af_err af_qr(af_array *q, af_array *r, af_array *tau, const af_array in);

    AFAPI af_err af_qr_inplace(af_array *tau, af_array in);

    AFAPI af_err af_cholesky(af_array *out, int *info, const af_array in, const bool is_upper);

    AFAPI af_err af_cholesky_inplace(int *info, af_array in, const bool is_upper);

    AFAPI af_err af_solve(af_array *out, const af_array a, const af_array b,
                          const af_solve_t options);

    AFAPI af_err af_inverse(af_array *out, const af_array in);

#ifdef __cplusplus
}
#endif
