/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/index.h>
#include "symbol_manager.hpp"

AFAPI af_err af_index(  af_array *out,
        const af_array in,
        const unsigned ndims, const af_seq* const index)
{
    return CALL(out, in, ndims, index);
}

AFAPI af_err af_lookup( af_array *out,
        const af_array in, const af_array indices,
        const unsigned dim)
{
    return CALL(out, in, indices, dim);
}

AFAPI af_err af_assign_seq( af_array *out,
        const af_array lhs,
        const unsigned ndims, const af_seq* const indices,
        const af_array rhs)
{
    return CALL(out, lhs, ndims, indices, rhs);
}

AFAPI af_err af_index_gen(  af_array *out,
        const af_array in,
        const dim_t ndims, const af_index_t* indices)
{
    return CALL(out, in, ndims, indices);
}

AFAPI af_err af_assign_gen( af_array *out,
        const af_array lhs,
        const dim_t ndims, const af_index_t* indices,
        const af_array rhs)
{
    return CALL(out, lhs, ndims, indices, rhs);
}
