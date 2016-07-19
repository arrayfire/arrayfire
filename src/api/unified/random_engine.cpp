/*******************************************************
 * Copyright(c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/random_engine.h>
#include "symbol_manager.hpp"

af_err af_create_random_engine(af_random_engine *engineHandle, af_random_type rtype, unsigned long long seed)
{
    return CALL(engineHandle, rtype, seed);
}

af_err af_random_engine_uniform(af_array *arr, af_random_engine engine, const unsigned ndims, const dim_t * const dims, const af_dtype type)
{
    return CALL(arr, engine, ndims, dims, type);
}

af_err af_random_engine_normal(af_array *arr, af_random_engine engine, const unsigned ndims, const dim_t * const dims, const af_dtype type)
{
    return CALL(arr, engine, ndims, dims, type);
}

af_err af_release_random_engine(af_random_engine engineHandle)
{
    return CALL(engineHandle);
}
