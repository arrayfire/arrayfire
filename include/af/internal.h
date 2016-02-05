/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>
#include <af/dim4.hpp>

#ifdef __cplusplus
namespace af
{
    class array;

    AFAPI array createArray(const void *data, const dim_t offset,
                            const dim4 dims, const dim4 strides,
                            const af::dtype ty,
                            const af::source location);

    AFAPI dim4 getStrides(const array &in);

    AFAPI dim_t getOffset(const array &in);

    AFAPI void *getRawPtr(const array &in);

    AFAPI bool isLinear(const array &in);

    AFAPI bool isOwner(const array &in);
}
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    AFAPI af_err af_create_array_with_strides(af_array *arr,
                                              const void *data,
                                              const dim_t offset,
                                              const unsigned ndims,
                                              const dim_t *const dims,
                                              const dim_t *const strides,
                                              const af_dtype ty,
                                              const af_source location);

    AFAPI af_err af_get_strides(dim_t *s0, dim_t *s1, dim_t *s2, dim_t *s3, const af_array arr);

    AFAPI af_err af_get_offset(dim_t *offset, const af_array arr);

    AFAPI af_err af_get_raw_ptr(void **ptr, const af_array arr);

    AFAPI af_err af_is_linear(bool *result, const af_array arr);

    AFAPI af_err af_is_owner(bool *result, const af_array arr);

#ifdef __cplusplus
}
#endif
