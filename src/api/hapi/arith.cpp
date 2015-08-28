/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/arith.h>
#include "symbol_manager.hpp"

#define BINARY_HAPI_DEF(af_func) \
af_err af_func(af_array* out, const af_array lhs, const af_array rhs, const bool batchMode) \
{ \
    return CALL(out, lhs, rhs, batchMode); \
}

BINARY_HAPI_DEF(af_add)
BINARY_HAPI_DEF(af_mul)
BINARY_HAPI_DEF(af_sub)
BINARY_HAPI_DEF(af_div)
BINARY_HAPI_DEF(af_maxof)
BINARY_HAPI_DEF(af_minof)
BINARY_HAPI_DEF(af_rem)
BINARY_HAPI_DEF(af_mod)
BINARY_HAPI_DEF(af_pow)
BINARY_HAPI_DEF(af_root)
BINARY_HAPI_DEF(af_atan2)
BINARY_HAPI_DEF(af_eq)
BINARY_HAPI_DEF(af_neq)
BINARY_HAPI_DEF(af_gt)
BINARY_HAPI_DEF(af_ge)
BINARY_HAPI_DEF(af_lt)
BINARY_HAPI_DEF(af_le)
BINARY_HAPI_DEF(af_and)
BINARY_HAPI_DEF(af_or)
BINARY_HAPI_DEF(af_bitand)
BINARY_HAPI_DEF(af_bitor)
BINARY_HAPI_DEF(af_bitxor)
BINARY_HAPI_DEF(af_bitshiftl)
BINARY_HAPI_DEF(af_bitshiftr)
