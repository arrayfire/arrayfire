/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/internal.h>
#include "symbol_manager.hpp"

af_err af_create_strided_array(af_array *arr, const void *data,
                               const dim_t offset, const unsigned ndims,
                               const dim_t *const dims_,
                               const dim_t *const strides_, const af_dtype ty,
                               const af_source location) {
    CALL(af_create_strided_array, arr, data, offset, ndims, dims_, strides_, ty,
         location);
}

af_err af_get_strides(dim_t *s0, dim_t *s1, dim_t *s2, dim_t *s3,
                      const af_array in) {
    CHECK_ARRAYS(in);
    CALL(af_get_strides, s0, s1, s2, s3, in);
}

af_err af_get_offset(dim_t *offset, const af_array arr) {
    CHECK_ARRAYS(arr);
    CALL(af_get_offset, offset, arr);
}

af_err af_get_raw_ptr(void **ptr, const af_array arr) {
    CHECK_ARRAYS(arr);
    CALL(af_get_raw_ptr, ptr, arr);
}

af_err af_is_linear(bool *result, const af_array arr) {
    CHECK_ARRAYS(arr);
    CALL(af_is_linear, result, arr);
}

af_err af_is_owner(bool *result, const af_array arr) {
    CHECK_ARRAYS(arr);
    CALL(af_is_owner, result, arr);
}

af_err af_get_allocated_bytes(size_t *bytes, const af_array arr) {
    CHECK_ARRAYS(arr);
    CALL(af_get_allocated_bytes, bytes, arr);
}
