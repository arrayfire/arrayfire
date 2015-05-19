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

#ifdef __cplusplus
namespace af
{
    class array;

    AFAPI void print(const char *exp, const array &arr);

    // Purpose of Addition: "How to add Function" documentation
    AFAPI array exampleFunction(const array& in, const af_someenum_t param);
}

#define af_print(exp) af::print(#exp, exp);

#endif //__cplusplus

#ifdef __cplusplus
extern "C" {
#endif

    // Get the number of elements in an af_array
    AFAPI af_err af_get_elements(dim_t *elems, const af_array arr);

    // Get the data type of an af_array
    AFAPI af_err af_get_type(af_dtype *type, const af_array arr);

    AFAPI af_err af_get_dims(dim_t *d0, dim_t *d1, dim_t *d2, dim_t *d3,
                             const af_array arr);

    AFAPI af_err af_get_numdims(unsigned *result, const af_array arr);

    AFAPI af_err af_is_empty        (bool *result, const af_array arr);

    AFAPI af_err af_is_scalar       (bool *result, const af_array arr);

    AFAPI af_err af_is_row          (bool *result, const af_array arr);

    AFAPI af_err af_is_column       (bool *result, const af_array arr);

    AFAPI af_err af_is_vector       (bool *result, const af_array arr);

    AFAPI af_err af_is_complex      (bool *result, const af_array arr);

    AFAPI af_err af_is_real         (bool *result, const af_array arr);

    AFAPI af_err af_is_double       (bool *result, const af_array arr);

    AFAPI af_err af_is_single       (bool *result, const af_array arr);

    AFAPI af_err af_is_realfloating (bool *result, const af_array arr);

    AFAPI af_err af_is_floating     (bool *result, const af_array arr);

    AFAPI af_err af_is_integer      (bool *result, const af_array arr);

    AFAPI af_err af_is_bool         (bool *result, const af_array arr);

    // Print contents of af_array to console
    AFAPI af_err af_print_array(af_array arr);

    // Purpose of Addition: "How to add Function" documentation
    AFAPI af_err af_example_function(af_array* out, const af_array in, const af_someenum_t param);

#ifdef __cplusplus
}
#endif
