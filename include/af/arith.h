#pragma once
#include "array.h"

#ifdef __cplusplus
extern "C" {
#endif

    AFAPI af_err af_add(af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_sub(af_array *result, const af_array lhs, const af_array rhs);
    AFAPI af_err af_mul(af_array *result, const af_array lhs, const af_array rhs);
#ifdef __cplusplus
}
#endif
