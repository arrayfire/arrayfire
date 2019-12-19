/*******************************************************
 * Copyright(c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/backend.h>
#include "symbol_manager.hpp"

af_err af_create_array(af_array *arr, const void *const data,
                       const unsigned ndims, const dim_t *const dims,
                       const af_dtype type) {
    CALL(af_create_array, arr, data, ndims, dims, type);
}

af_err af_create_handle(af_array *arr, const unsigned ndims,
                        const dim_t *const dims, const af_dtype type) {
    CALL(af_create_handle, arr, ndims, dims, type);
}

af_err af_copy_array(af_array *arr, const af_array in) {
    CHECK_ARRAYS(in);
    CALL(af_copy_array, arr, in);
}

af_err af_write_array(af_array arr, const void *data, const size_t bytes,
                      af_source src) {
    CHECK_ARRAYS(arr);
    CALL(af_write_array, arr, data, bytes, src);
}

af_err af_get_data_ptr(void *data, const af_array arr) {
    CHECK_ARRAYS(arr);
    CALL(af_get_data_ptr, data, arr);
}

af_err af_release_array(af_array arr) {
    if (arr) {
        CALL(af_release_array, arr);
    } else {
        return AF_SUCCESS;
    }
}

af_err af_retain_array(af_array *out, const af_array in) {
    CHECK_ARRAYS(in);
    CALL(af_retain_array, out, in);
}

af_err af_get_data_ref_count(int *use_count, const af_array in) {
    CHECK_ARRAYS(in);
    CALL(af_get_data_ref_count, use_count, in);
}

af_err af_eval(af_array in) {
    CHECK_ARRAYS(in);
    CALL(af_eval, in);
}

af_err af_get_elements(dim_t *elems, const af_array arr) {
    CHECK_ARRAYS(arr);
    CALL(af_get_elements, elems, arr);
}

af_err af_get_type(af_dtype *type, const af_array arr) {
    CHECK_ARRAYS(arr);
    CALL(af_get_type, type, arr);
}

af_err af_get_dims(dim_t *d0, dim_t *d1, dim_t *d2, dim_t *d3,
                   const af_array arr) {
    CHECK_ARRAYS(arr);
    CALL(af_get_dims, d0, d1, d2, d3, arr);
}

af_err af_get_numdims(unsigned *result, const af_array arr) {
    CHECK_ARRAYS(arr);
    CALL(af_get_numdims, result, arr);
}

#define ARRAY_HAPI_DEF(af_func)                        \
    af_err af_func(bool *result, const af_array arr) { \
        CHECK_ARRAYS(arr);                             \
        CALL(af_func, result, arr);                    \
    }

ARRAY_HAPI_DEF(af_is_empty)
ARRAY_HAPI_DEF(af_is_scalar)
ARRAY_HAPI_DEF(af_is_row)
ARRAY_HAPI_DEF(af_is_column)
ARRAY_HAPI_DEF(af_is_vector)
ARRAY_HAPI_DEF(af_is_complex)
ARRAY_HAPI_DEF(af_is_real)
ARRAY_HAPI_DEF(af_is_double)
ARRAY_HAPI_DEF(af_is_single)
ARRAY_HAPI_DEF(af_is_half)
ARRAY_HAPI_DEF(af_is_realfloating)
ARRAY_HAPI_DEF(af_is_floating)
ARRAY_HAPI_DEF(af_is_integer)
ARRAY_HAPI_DEF(af_is_bool)
ARRAY_HAPI_DEF(af_is_sparse)

af_err af_get_scalar(void *output_value, const af_array arr) {
    CHECK_ARRAYS(arr);
    CALL(af_get_scalar, output_value, arr);
}
