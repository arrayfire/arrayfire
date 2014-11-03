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
    class array;

    AFAPI array min    (const array &lhs, const array &rhs);
    AFAPI array max    (const array &lhs, const array &rhs);

    AFAPI array cplx2  (const array &lhs, const array &rhs);
    AFAPI array atan2  (const array &lhs, const array &rhs);

    AFAPI array pow    (const array &lhs, const array &rhs);
    AFAPI array pow    (const array &lhs, const double rhs);
    AFAPI array rem    (const array &lhs, const array &rhs);
    AFAPI array mod    (const array &lhs, const array &rhs);

    AFAPI array cplx   (const array &in);
    AFAPI array abs    (const array &in);

    AFAPI array round  (const array &in);
    AFAPI array floor  (const array &in);
    AFAPI array ceil   (const array &in);

    AFAPI array sin    (const array &in);
    AFAPI array cos    (const array &in);
    AFAPI array tan    (const array &in);

    AFAPI array asin   (const array &in);
    AFAPI array acos   (const array &in);
    AFAPI array atan   (const array &in);

    AFAPI array sinh   (const array &in);
    AFAPI array cosh   (const array &in);
    AFAPI array tanh   (const array &in);

    AFAPI array asinh  (const array &in);
    AFAPI array acosh  (const array &in);
    AFAPI array atanh  (const array &in);

    AFAPI array exp    (const array &in);
    AFAPI array expm1  (const array &in);
    AFAPI array erf    (const array &in);
    AFAPI array erfc   (const array &in);

    AFAPI array log    (const array &in);
    AFAPI array log1p  (const array &in);
    AFAPI array log10  (const array &in);

    AFAPI array sqrt   (const array &in);
    AFAPI array cbrt   (const array &in);

    AFAPI array iszero (const array &in);
    AFAPI array isInf  (const array &in);
    AFAPI array isNaN  (const array &in);

    AFAPI array tgamma (const array &in);
    AFAPI array lgamma (const array &in);
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    AFAPI af_err af_add   (af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_sub   (af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_mul   (af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_div   (af_array *result, const af_array lhs, const af_array rhs);

    AFAPI af_err af_lt    (af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_gt    (af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_le    (af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_ge    (af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_eq    (af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_neq   (af_array *result, const af_array lhs, const af_array rhs);

    AFAPI af_err af_minof (af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_maxof (af_array *result, const af_array lhs, const af_array rhs);

    AFAPI af_err af_cplx2 (af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_atan2 (af_array *result, const af_array lhs, const af_array rhs);

    AFAPI af_err af_pow   (af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_rem   (af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_mod   (af_array *result, const af_array lhs, const af_array rhs);

    AFAPI af_err af_cast    (af_array *out, const af_array in, af_dtype type);
    AFAPI af_err af_cplx    (af_array *out, const af_array in);
    AFAPI af_err af_abs     (af_array *out, const af_array in);

    AFAPI af_err af_round   (af_array *out, const af_array in);
    AFAPI af_err af_floor   (af_array *out, const af_array in);
    AFAPI af_err af_ceil    (af_array *out, const af_array in);

    AFAPI af_err af_sin     (af_array *out, const af_array in);
    AFAPI af_err af_cos     (af_array *out, const af_array in);
    AFAPI af_err af_tan     (af_array *out, const af_array in);

    AFAPI af_err af_asin    (af_array *out, const af_array in);
    AFAPI af_err af_acos    (af_array *out, const af_array in);
    AFAPI af_err af_atan    (af_array *out, const af_array in);

    AFAPI af_err af_sinh    (af_array *out, const af_array in);
    AFAPI af_err af_cosh    (af_array *out, const af_array in);
    AFAPI af_err af_tanh    (af_array *out, const af_array in);

    AFAPI af_err af_asinh   (af_array *out, const af_array in);
    AFAPI af_err af_acosh   (af_array *out, const af_array in);
    AFAPI af_err af_atanh   (af_array *out, const af_array in);

    AFAPI af_err af_exp     (af_array *out, const af_array in);
    AFAPI af_err af_expm1   (af_array *out, const af_array in);
    AFAPI af_err af_erf     (af_array *out, const af_array in);
    AFAPI af_err af_erfc    (af_array *out, const af_array in);

    AFAPI af_err af_log     (af_array *out, const af_array in);
    AFAPI af_err af_log1p   (af_array *out, const af_array in);
    AFAPI af_err af_log10   (af_array *out, const af_array in);

    AFAPI af_err af_sqrt    (af_array *out, const af_array in);
    AFAPI af_err af_cbrt    (af_array *out, const af_array in);

    AFAPI af_err af_iszero  (af_array *out, const af_array in);
    AFAPI af_err af_isinf   (af_array *out, const af_array in);
    AFAPI af_err af_isnan   (af_array *out, const af_array in);

    AFAPI af_err af_tgamma   (af_array *out, const af_array in);
    AFAPI af_err af_lgamma   (af_array *out, const af_array in);

#ifdef __cplusplus
}
#endif
