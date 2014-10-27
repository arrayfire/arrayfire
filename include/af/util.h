#pragma once
#include <af/defines.h>

namespace af
{
    AFAPI void print(const char *exp, const array arr);
}

#define af_print(exp) af::print(#exp, exp);

#ifdef __cplusplus
extern "C" {
#endif

    // Get the number of elements in an af_array
    AFAPI af_err af_get_elements(dim_type *elems, const af_array arr);

    // Get the data type of an af_array
    AFAPI af_err af_get_type(af_dtype *type, const af_array arr);

    // Print contents of af_array to console
    AFAPI af_err af_print_array(af_array arr);

#ifdef __cplusplus
}
#endif
