/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/internal.h>
#include "error.hpp"

namespace af {
array createStridedArray(
    const void *data, const dim_t offset,
    const dim4 dims,     // NOLINT(performance-unnecessary-value-param)
    const dim4 strides,  // NOLINT(performance-unnecessary-value-param)
    const af::dtype ty, const af::source location) {
    af_array res;
    AF_THROW(af_create_strided_array(&res, data, offset, dims.ndims(),
                                     dims.get(), strides.get(), ty, location));
    return array(res);
}

dim4 getStrides(const array &in) {
    dim_t s0, s1, s2, s3;
    AF_THROW(af_get_strides(&s0, &s1, &s2, &s3, in.get()));
    return dim4(s0, s1, s2, s3);
}

dim_t getOffset(const array &in) {
    dim_t offset;
    AF_THROW(af_get_offset(&offset, in.get()));
    return offset;
}

void *getRawPtr(const array &in) {
    void *ptr = NULL;
    AF_THROW(af_get_raw_ptr(&ptr, in.get()));
    return ptr;
}

bool isLinear(const array &in) {
    bool is_linear = false;
    AF_THROW(af_is_linear(&is_linear, in.get()));
    return is_linear;
}

bool isOwner(const array &in) {
    bool is_owner = false;
    AF_THROW(af_is_owner(&is_owner, in.get()));
    return is_owner;
}

}  // namespace af
