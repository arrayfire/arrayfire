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
    return CALL(arr, data, offset, ndims, dims_, strides_, ty, location);
}

af_err af_get_strides(dim_t *s0, dim_t *s1, dim_t *s2, dim_t *s3,
                      const af_array in) {
    CHECK_ARRAYS(in);
    return CALL(s0, s1, s2, s3, in);
}

af_err af_get_offset(dim_t *offset, const af_array arr) {
    CHECK_ARRAYS(arr);
    return CALL(offset, arr);
}

af_err af_get_raw_ptr(void **ptr, const af_array arr) {
    CHECK_ARRAYS(arr);
    return CALL(ptr, arr);
}

af_err af_is_linear(bool *result, const af_array arr) {
    CHECK_ARRAYS(arr);
    return CALL(result, arr);
}

af_err af_is_owner(bool *result, const af_array arr) {
    CHECK_ARRAYS(arr);
    return CALL(result, arr);
}

af_err af_get_allocated_bytes(size_t *bytes, const af_array arr) {
    CHECK_ARRAYS(arr);
    return CALL(bytes, arr);
}

af_err af_set_use_events_based_memory_manager(int enable) {
    return CALL(enable);
}
