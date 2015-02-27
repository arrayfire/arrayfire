/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/defines.h>
#include <af/arith.h>
#include <af/data.h>
#include <ArrayInfo.hpp>
#include <optypes.hpp>
#include <implicit.hpp>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <unary.hpp>
#include <implicit.hpp>

using namespace detail;

template<typename T, af_op_t op>
static inline af_array unaryOp(const af_array in)
{
    af_array res = getHandle(unaryOp<T, op>(getArray<T>(in)));
    // All inputs to this function are temporary references
    // Delete the temporary references
    destroyHandle<T>(in);
    return res;
}

template<af_op_t op>
static af_err af_unary(af_array *out, const af_array in)
{
    try {

        ArrayInfo in_info = getInfo(in);
        ARG_ASSERT(1, in_info.isReal());

        af_dtype in_type = in_info.getType();
        af_array res;

        // Convert all inputs to floats / doubles
        af_dtype type = implicit(in_type, f32);
        af_array input = cast(in, type);

        switch (type) {
        case f32 : res = unaryOp<float  , op>(input); break;
        case f64 : res = unaryOp<double , op>(input); break;
        default:
            TYPE_ERROR(1, in_type); break;
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

#define UNARY(fn)                                       \
    af_err af_##fn(af_array *out, const af_array in)    \
    {                                                   \
        return af_unary<af_##fn##_t>(out, in);          \
    }


UNARY(sin)
UNARY(cos)
UNARY(tan)

UNARY(asin)
UNARY(acos)
UNARY(atan)

UNARY(sinh)
UNARY(cosh)
UNARY(tanh)

UNARY(asinh)
UNARY(acosh)
UNARY(atanh)

UNARY(round)
UNARY(floor)
UNARY(ceil)

UNARY(exp)
UNARY(expm1)
UNARY(erf)
UNARY(erfc)

UNARY(log)
UNARY(log10)
UNARY(log1p)

UNARY(sqrt)
UNARY(cbrt)

UNARY(tgamma)
UNARY(lgamma)

af_err af_not(af_array *out, const af_array in)
{
    try {

        af_array tmp;
        ArrayInfo in_info = getInfo(in);

        AF_CHECK(af_constant(&tmp, 0,
                             in_info.ndims(),
                             in_info.dims().get(), in_info.getType()));

        AF_CHECK(af_neq(out, in, tmp, false));

        AF_CHECK(af_destroy_array(tmp));
    } CATCHALL;

    return AF_SUCCESS;
}

template<typename T, af_op_t op>
static inline af_array checkOp(const af_array in)
{
    af_array res = getHandle(checkOp<T, op>(getArray<T>(in)));
    // All inputs to this function are temporary references
    // Delete the temporary references
    destroyHandle<T>(in);
    return res;
}

template<af_op_t op>
static af_err af_check(af_array *out, const af_array in)
{
    try {

        ArrayInfo in_info = getInfo(in);
        ARG_ASSERT(1, in_info.isReal());

        af_dtype in_type = in_info.getType();
        af_array res;

        // Convert all inputs to floats / doubles
        af_dtype type = implicit(in_type, f32);
        af_array input = cast(in, type);

        switch (type) {
        case f32 : res = checkOp<float  , op>(input); break;
        case f64 : res = checkOp<double , op>(input); break;
        default:
            TYPE_ERROR(1, in_type); break;
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

#define CHECK(fn)                                       \
    af_err af_##fn(af_array *out, const af_array in)    \
    {                                                   \
        return af_check<af_##fn##_t>(out, in);          \
    }


CHECK(isinf)
CHECK(isnan)
CHECK(iszero)
