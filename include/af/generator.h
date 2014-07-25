#pragma once
#include <af/defines.h>
#include <af/array.h>

#ifdef __cplusplus
extern "C" {
#endif

    AFAPI af_err af_create_array(af_array *arr, const unsigned ndims, const long * const dims, af_dtype type);
    AFAPI af_err af_constant(af_array *arr, double val, const unsigned ndims, const long * const dims, af_dtype type);

#ifdef __cplusplus
}
#endif
