/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/arith.h>
#include <af/array.h>
#include "symbol_manager.hpp"

#define BINARY_HAPI_DEF(af_func)                                          \
    af_err af_func(af_array* out, const af_array lhs, const af_array rhs, \
                   const bool batchMode) {                                \
        CHECK_ARRAYS(lhs, rhs);                                           \
        CALL(af_func, out, lhs, rhs, batchMode);                          \
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
BINARY_HAPI_DEF(af_cplx2)
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
BINARY_HAPI_DEF(af_hypot)

af_err af_cast(af_array* out, const af_array in, const af_dtype type) {
    CHECK_ARRAYS(in);
    CALL(af_cast, out, in, type);
}

#define UNARY_HAPI_DEF(af_func)                        \
    af_err af_func(af_array* out, const af_array in) { \
        CHECK_ARRAYS(in);                              \
        CALL(af_func, out, in);                        \
    }

UNARY_HAPI_DEF(af_abs)
UNARY_HAPI_DEF(af_arg)
UNARY_HAPI_DEF(af_sign)
UNARY_HAPI_DEF(af_round)
UNARY_HAPI_DEF(af_trunc)
UNARY_HAPI_DEF(af_floor)
UNARY_HAPI_DEF(af_ceil)
UNARY_HAPI_DEF(af_sin)
UNARY_HAPI_DEF(af_cos)
UNARY_HAPI_DEF(af_tan)
UNARY_HAPI_DEF(af_asin)
UNARY_HAPI_DEF(af_acos)
UNARY_HAPI_DEF(af_atan)
UNARY_HAPI_DEF(af_cplx)
UNARY_HAPI_DEF(af_real)
UNARY_HAPI_DEF(af_imag)
UNARY_HAPI_DEF(af_conjg)
UNARY_HAPI_DEF(af_sinh)
UNARY_HAPI_DEF(af_cosh)
UNARY_HAPI_DEF(af_tanh)
UNARY_HAPI_DEF(af_asinh)
UNARY_HAPI_DEF(af_acosh)
UNARY_HAPI_DEF(af_atanh)
UNARY_HAPI_DEF(af_pow2)
UNARY_HAPI_DEF(af_exp)
UNARY_HAPI_DEF(af_sigmoid)
UNARY_HAPI_DEF(af_expm1)
UNARY_HAPI_DEF(af_erf)
UNARY_HAPI_DEF(af_erfc)
UNARY_HAPI_DEF(af_log)
UNARY_HAPI_DEF(af_log1p)
UNARY_HAPI_DEF(af_log10)
UNARY_HAPI_DEF(af_log2)
UNARY_HAPI_DEF(af_sqrt)
UNARY_HAPI_DEF(af_rsqrt)
UNARY_HAPI_DEF(af_cbrt)
UNARY_HAPI_DEF(af_factorial)
UNARY_HAPI_DEF(af_tgamma)
UNARY_HAPI_DEF(af_lgamma)
UNARY_HAPI_DEF(af_iszero)
UNARY_HAPI_DEF(af_isinf)
UNARY_HAPI_DEF(af_isnan)
UNARY_HAPI_DEF(af_not)
UNARY_HAPI_DEF(af_bitnot)

af_err af_clamp(af_array* out, const af_array in, const af_array lo,
                const af_array hi, const bool batch) {
    CHECK_ARRAYS(in, lo, hi);
    CALL(af_clamp, out, in, lo, hi, batch);
}
