#pragma once
#include "array.h"

#ifdef __cplusplus
extern "C" {
#endif

    AFAPI af_err af_add(af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_sub(af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_mul(af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_div(af_array *result, const af_array lhs, const af_array rhs);

    AFAPI af_err af_lt(af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_gt(af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_le(af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_ge(af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_eq(af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_neq(af_array *result, const af_array lhs, const af_array rhs);

#ifdef __cplusplus
}
#endif
