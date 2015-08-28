/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/data.h>
#include "symbol_manager.hpp"

af_err af_create_array(af_array *result, const void * const data,
                       const unsigned ndims, const dim_t * const dims,
                       const af_dtype type)
{
    return CALL(result, data, ndims, dims, type);
}

af_err af_constant(af_array *result, const double value,
                   const unsigned ndims, const dim_t * const dims,
                   const af_dtype type)
{
    return CALL(result, value, ndims, dims, type);
}

af_err af_release_array(af_array arr)
{
    return CALL(arr);
}
