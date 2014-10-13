#pragma once
#include "array.h"

#ifdef __cplusplus
extern "C" {
#endif

    AFAPI af_err af_add   (af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_sub   (af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_mul   (af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_div   (af_array *result, const af_array lhs, const af_array rhs);

    AFAPI af_err af_minof (af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_maxof (af_array *result, const af_array lhs, const af_array rhs);

    AFAPI af_err af_lt    (af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_gt    (af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_le    (af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_ge    (af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_eq    (af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_neq   (af_array *result, const af_array lhs, const af_array rhs);

    AFAPI af_err af_cplx2 (af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_atan2 (af_array *result, const af_array lhs, const af_array rhs);

    AFAPI af_err af_pow   (af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_rem   (af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_mod   (af_array *result, const af_array lhs, const af_array rhs);

    AFAPI af_err af_cast    (af_array *out, const af_array in);
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
