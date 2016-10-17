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

#if AF_API_VERSION >= 33
    /**
       \param[in] data is the raw data pointer.
       \param[in] offset specifies the number of elements to skip.
       \param[in] dims specifies the dimensions for the region of interest.
       \param[in] strides specifies the distance between each element of a given dimension.
       \param[in] ty specifies the data type of \p data.
       \param[in] location specifies if the data is on host or the device.

       \note: If \p location is `afHost`, a memory copy is performed.

       \returns an af::array() with specified offset, dimensions and strides.

       \ingroup internal_func_create
    */
    AFAPI array createStridedArray(const void *data, const dim_t offset,
                                   const dim4 dims, const dim4 strides,
                                   const af::dtype ty,
                                   const af::source location);
#endif

#if AF_API_VERSION >= 33
    /**
       \param[in] in An multi dimensional array.
       \returns af::dim4() containing distance between consecutive elements in each dimension.

       \ingroup internal_func_strides
    */
    AFAPI dim4 getStrides(const array &in);
#endif

#if AF_API_VERSION >= 33
    /**
       \param[in] in An multi dimensional array.
       \returns offset from the starting location of data pointer specified in number of elements.

       \ingroup internal_func_offset
    */
    AFAPI dim_t getOffset(const array &in);
#endif

#if AF_API_VERSION >= 33
    /**
       \param[in] in An multi dimensional array.
       \returns Returns the raw pointer location to the array.

       \note This pointer may be shared with other arrays. Use this function with caution.

       \ingroup internal_func_rawptr
    */
    AFAPI void *getRawPtr(const array &in);
#endif

#if AF_API_VERSION >= 33
    /**
       \param[in] in An multi dimensional array.
       \returns a boolean specifying if all elements in the array are contiguous.

       \ingroup internal_func_linear
    */
    AFAPI bool isLinear(const array &in);
#endif

#if AF_API_VERSION >= 33
    /**
       \param[in] in An multi dimensional array.
       \returns a boolean specifying if the array owns the raw pointer. It is false if it is a sub array.

       \ingroup internal_func_owner
    */
    AFAPI bool isOwner(const array &in);
#endif
}
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#if AF_API_VERSION >= 33
    /**
       \param[out] arr an af_array with specified offset, dimensions and strides.
       \param[in] data is the raw data pointer.
       \param[in] offset specifies the number of elements to skip.
       \param[in] ndims specifies the number of array dimensions.
       \param[in] dims specifies the dimensions for the region of interest.
       \param[in] strides specifies the distance between each element of a given dimension.
       \param[in] ty specifies the data type of \p data.
       \param[in] location specifies if the data is on host or the device.

       \note If \p location is `afHost`, a memory copy is performed.

       \ingroup internal_func_create
    */
    AFAPI af_err af_create_strided_array(af_array *arr,
                                         const void *data,
                                         const dim_t offset,
                                         const unsigned ndims,
                                         const dim_t *const dims,
                                         const dim_t *const strides,
                                         const af_dtype ty,
                                         const af_source location);
#endif

#if AF_API_VERSION >= 33
    /**
       \param[in] arr An multi dimensional array.
       \param[out] s0 distance between each consecutive element along first  dimension.
       \param[out] s1 distance between each consecutive element along second dimension.
       \param[out] s2 distance between each consecutive element along third  dimension.
       \param[out] s3 distance between each consecutive element along fourth dimension.

       \ingroup internal_func_strides
    */
    AFAPI af_err af_get_strides(dim_t *s0, dim_t *s1, dim_t *s2, dim_t *s3, const af_array arr);
#endif

#if AF_API_VERSION >= 33
    /**
       \param[in] arr An multi dimensional array.
       \param[out] offset: Offset from the starting location of data pointer specified in number of elements. distance between each consecutive element along first  dimension.

       \ingroup internal_func_offset
    */
    AFAPI af_err af_get_offset(dim_t *offset, const af_array arr);
#endif

#if AF_API_VERSION >= 33
    /**
       \param[in] arr An multi dimensional array.
       \param[out] ptr the raw pointer location to the array.

       \note This pointer may be shared with other arrays. Use this function with caution.

       \ingroup internal_func_rawptr
    */
    AFAPI af_err af_get_raw_ptr(void **ptr, const af_array arr);
#endif

#if AF_API_VERSION >= 33
    /**
       \param[in] arr An multi dimensional array.
       \param[out] result: a boolean specifying if all elements in the array are contiguous.

       \ingroup internal_func_linear
    */
    AFAPI af_err af_is_linear(bool *result, const af_array arr);
#endif

#if AF_API_VERSION >= 33
    /**
       \param[in] arr An multi dimensional array.
       \param[out] result: a boolean specifying if the array owns the raw pointer. It is false if it is a sub array.

       \ingroup internal_func_owner
    */
    AFAPI af_err af_is_owner(bool *result, const af_array arr);
#endif

#if AF_API_VERSION >= 35
    /**
       \param[out] bytes the size of the physical allocated bytes. This will return the size
       of the parent/owner if the \p arr is an indexed array.
       \param[in] arr the input array.

       \ingroup internal_func_allocatedbytes
    */
    AFAPI af_err af_get_allocated_bytes(size_t *bytes, const af_array arr);
#endif

#ifdef __cplusplus
}
#endif
