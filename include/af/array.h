#pragma once
#include <af/defines.h>
#include <af/dim4.hpp>
#define AF_MAX_DIMS 4

typedef long long af_array;

#ifdef __cplusplus
extern "C" {
#endif

    AFAPI dim_type  af_get_elements(af_array arr);
    AFAPI af_dtype  af_get_type(af_array arr);

    AFAPI af_err af_host_ptr(void **ptr, af_array arr);
    AFAPI af_err af_copy(af_array *dst, const void* const src);

    AFAPI af_err af_print(af_array arr);

    AFAPI af_err af_index(af_array *out, af_array in, unsigned ndims, const af_seq* const index );
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
namespace af {
    class array {
    };
}
#endif
