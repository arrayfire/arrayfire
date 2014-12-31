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

#include <arith.hpp>
#include <logic.hpp>

using namespace detail;
using af::dim4;

static dim4 getOutDims(const dim4 ldims, const dim4 rdims, bool batchMode)
{
    if (!batchMode) {
        DIM_ASSERT(1, ldims == rdims);
        return ldims;
    }

    AF_ERROR("Batch mode not supported yet", AF_ERR_NOT_SUPPORTED);
}

template<typename T, af_op_t op>
static inline af_array arithOp(const af_array lhs, const af_array rhs,
                               const dim4 &odims)
{
    af_array res = getHandle(*arithOp<T, op>(getArray<T>(lhs), getArray<T>(rhs), odims));
    // All inputs to this function are temporary references
    // Delete the temporary references
    destroyHandle<T>(lhs);
    destroyHandle<T>(rhs);
    return res;
}

template<af_op_t op>
static af_err af_arith(af_array *out, const af_array lhs, const af_array rhs, bool batchMode)
{
    try {
        const af_dtype otype = implicit(lhs, rhs);
        const af_array left  = cast(lhs, otype);
        const af_array right = cast(rhs, otype);

        ArrayInfo linfo = getInfo(lhs);
        ArrayInfo rinfo = getInfo(rhs);

        dim4 odims = getOutDims(linfo.dims(), rinfo.dims(), batchMode);

        af_array res;
        switch (otype) {
        case f32: res = arithOp<float  , op>(left, right, odims); break;
        case f64: res = arithOp<double , op>(left, right, odims); break;
        case c32: res = arithOp<cfloat , op>(left, right, odims); break;
        case c64: res = arithOp<cdouble, op>(left, right, odims); break;
        case s32: res = arithOp<int    , op>(left, right, odims); break;
        case u32: res = arithOp<uint   , op>(left, right, odims); break;
        case u8 : res = arithOp<uchar  , op>(left, right, odims); break;
        case b8 : res = arithOp<char   , op>(left, right, odims); break;
        default: TYPE_ERROR(0, otype);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

template<af_op_t op>
static af_err af_arith_real(af_array *out, const af_array lhs, const af_array rhs, bool batchMode)
{
    try {
        const af_dtype otype = implicit(lhs, rhs);
        const af_array left  = cast(lhs, otype);
        const af_array right = cast(rhs, otype);

        ArrayInfo linfo = getInfo(lhs);
        ArrayInfo rinfo = getInfo(rhs);

        dim4 odims = getOutDims(linfo.dims(), rinfo.dims(), batchMode);

        af_array res;
        switch (otype) {
        case f32: res = arithOp<float  , op>(left, right, odims); break;
        case f64: res = arithOp<double , op>(left, right, odims); break;
        case s32: res = arithOp<int    , op>(left, right, odims); break;
        case u32: res = arithOp<uint   , op>(left, right, odims); break;
        case u8 : res = arithOp<uchar  , op>(left, right, odims); break;
        case b8 : res = arithOp<char   , op>(left, right, odims); break;
        default: TYPE_ERROR(0, otype);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_add(af_array *out, const af_array lhs, const af_array rhs, bool batchMode)
{
    return af_arith<af_add_t>(out, lhs, rhs, batchMode);
}

af_err af_mul(af_array *out, const af_array lhs, const af_array rhs, bool batchMode)
{
    return af_arith<af_mul_t>(out, lhs, rhs, batchMode);
}

af_err af_sub(af_array *out, const af_array lhs, const af_array rhs, bool batchMode)
{
    return af_arith<af_sub_t>(out, lhs, rhs, batchMode);
}

af_err af_div(af_array *out, const af_array lhs, const af_array rhs, bool batchMode)
{
    return af_arith<af_div_t>(out, lhs, rhs, batchMode);
}

af_err af_maxof(af_array *out, const af_array lhs, const af_array rhs, bool batchMode)
{
    return af_arith<af_max_t>(out, lhs, rhs, batchMode);
}

af_err af_minof(af_array *out, const af_array lhs, const af_array rhs, bool batchMode)
{
    return af_arith<af_min_t>(out, lhs, rhs, batchMode);
}

af_err af_rem(af_array *out, const af_array lhs, const af_array rhs, bool batchMode)
{
    return af_arith_real<af_rem_t>(out, lhs, rhs, batchMode);
}

af_err af_mod(af_array *out, const af_array lhs, const af_array rhs, bool batchMode)
{
    return af_arith_real<af_mod_t>(out, lhs, rhs, batchMode);
}

af_err af_pow(af_array *out, const af_array lhs, const af_array rhs, bool batchMode)
{
    try {
        ArrayInfo linfo = getInfo(lhs);
        ArrayInfo rinfo = getInfo(rhs);
        if (linfo.isComplex() || rinfo.isComplex()) {
            AF_ERROR("Powers of Complex numbers not supported", AF_ERR_NOT_SUPPORTED);
        }
        return af_arith<af_pow_t>(out, lhs, rhs, batchMode);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_atan2(af_array *out, const af_array lhs, const af_array rhs, bool batchMode)
{
    try {

        const af_dtype type = implicit(lhs, rhs);

        const af_array left  = cast(lhs, type);
        const af_array right = cast(rhs, type);

        if (type != f32 && type != f64) {
            AF_ERROR("Only floating point arrays are supported for atan2 ",
                     AF_ERR_NOT_SUPPORTED);
        }

        ArrayInfo linfo = getInfo(lhs);
        ArrayInfo rinfo = getInfo(rhs);

        dim4 odims = getOutDims(linfo.dims(), rinfo.dims(), batchMode);

        af_array res;
        switch (type) {
        case f32: res = arithOp<float , af_atan2_t>(left, right, odims); break;
        case f64: res = arithOp<double, af_atan2_t>(left, right, odims); break;
        default: TYPE_ERROR(0, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_hypot(af_array *out, const af_array lhs, const af_array rhs, bool batchMode)
{
    try {

        const af_dtype type = implicit(lhs, rhs);

        const af_array left  = cast(lhs, type);
        const af_array right = cast(rhs, type);

        if (type != f32 && type != f64) {
            AF_ERROR("Only floating point arrays are supported for hypot ",
                     AF_ERR_NOT_SUPPORTED);
        }

        ArrayInfo linfo = getInfo(lhs);
        ArrayInfo rinfo = getInfo(rhs);

        dim4 odims = getOutDims(linfo.dims(), rinfo.dims(), batchMode);

        af_array res;
        switch (type) {
        case f32: res = arithOp<float , af_hypot_t>(left, right, odims); break;
        case f64: res = arithOp<double, af_hypot_t>(left, right, odims); break;
        default: TYPE_ERROR(0, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

template<typename T, af_op_t op>
static inline af_array logicOp(const af_array lhs, const af_array rhs, const dim4 &odims)
{
    af_array res = getHandle(*logicOp<T, op>(getArray<T>(lhs), getArray<T>(rhs), odims));
    // All inputs to this function are temporary references
    // Delete the temporary references
    destroyHandle<T>(lhs);
    destroyHandle<T>(rhs);
    return res;
}

template<af_op_t op>
static af_err af_logic(af_array *out, const af_array lhs, const af_array rhs, bool batchMode)
{
    try {
        const af_dtype type = implicit(lhs, rhs);

        const af_array left  = cast(lhs, type);
        const af_array right = cast(rhs, type);

        ArrayInfo linfo = getInfo(lhs);
        ArrayInfo rinfo = getInfo(rhs);

        dim4 odims = getOutDims(linfo.dims(), rinfo.dims(), batchMode);

        af_array res;
        switch (type) {
        case f32: res = logicOp<float  , op>(left, right, odims); break;
        case f64: res = logicOp<double , op>(left, right, odims); break;
        case c32: res = logicOp<cfloat , op>(left, right, odims); break;
        case c64: res = logicOp<cdouble, op>(left, right, odims); break;
        case s32: res = logicOp<int    , op>(left, right, odims); break;
        case u32: res = logicOp<uint   , op>(left, right, odims); break;
        case u8 : res = logicOp<uchar  , op>(left, right, odims); break;
        case b8 : res = logicOp<char   , op>(left, right, odims); break;
        default: TYPE_ERROR(0, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_eq(af_array *out, const af_array lhs, const af_array rhs, bool batchMode)
{
    return af_logic<af_eq_t>(out, lhs, rhs, batchMode);
}

af_err af_neq(af_array *out, const af_array lhs, const af_array rhs, bool batchMode)
{
    return af_logic<af_neq_t>(out, lhs, rhs, batchMode);
}

af_err af_gt(af_array *out, const af_array lhs, const af_array rhs, bool batchMode)
{
    return af_logic<af_gt_t>(out, lhs, rhs, batchMode);
}

af_err af_ge(af_array *out, const af_array lhs, const af_array rhs, bool batchMode)
{
    return af_logic<af_ge_t>(out, lhs, rhs, batchMode);
}

af_err af_lt(af_array *out, const af_array lhs, const af_array rhs, bool batchMode)
{
    return af_logic<af_lt_t>(out, lhs, rhs, batchMode);
}

af_err af_le(af_array *out, const af_array lhs, const af_array rhs, bool batchMode)
{
    return af_logic<af_le_t>(out, lhs, rhs, batchMode);
}

af_err af_and(af_array *out, const af_array lhs, const af_array rhs, bool batchMode)
{
    return af_logic<af_and_t>(out, lhs, rhs, batchMode);
}

af_err af_or(af_array *out, const af_array lhs, const af_array rhs, bool batchMode)
{
    return af_logic<af_or_t>(out, lhs, rhs, batchMode);
}

af_err af_bitand(af_array *out, const af_array lhs, const af_array rhs, bool batchMode)
{
    return AF_ERR_NOT_SUPPORTED;
}

af_err af_bitor(af_array *out, const af_array lhs, const af_array rhs, bool batchMode)
{
    return AF_ERR_NOT_SUPPORTED;
}

af_err af_bitxor(af_array *out, const af_array lhs, const af_array rhs, bool batchMode)
{
    return AF_ERR_NOT_SUPPORTED;
}
