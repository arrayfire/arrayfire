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

template<typename T, af_op_t op>
static inline af_array arithOp(const af_array lhs, const af_array rhs,
                               const dim4 &odims)
{
    af_array res = getHandle(arithOp<T, op>(castArray<T>(lhs), castArray<T>(rhs), odims));
    return res;
}

template<af_op_t op>
static af_err af_arith(af_array *out, const af_array lhs, const af_array rhs, const bool batchMode)
{
    try {
        const af_dtype otype = implicit(lhs, rhs);

        ArrayInfo linfo = getInfo(lhs);
        ArrayInfo rinfo = getInfo(rhs);

        dim4 odims = getOutDims(linfo.dims(), rinfo.dims(), batchMode);

        af_array res;
        switch (otype) {
        case f32: res = arithOp<float  , op>(lhs, rhs, odims); break;
        case f64: res = arithOp<double , op>(lhs, rhs, odims); break;
        case c32: res = arithOp<cfloat , op>(lhs, rhs, odims); break;
        case c64: res = arithOp<cdouble, op>(lhs, rhs, odims); break;
        case s32: res = arithOp<int    , op>(lhs, rhs, odims); break;
        case u32: res = arithOp<uint   , op>(lhs, rhs, odims); break;
        case u8 : res = arithOp<uchar  , op>(lhs, rhs, odims); break;
        case b8 : res = arithOp<char   , op>(lhs, rhs, odims); break;
        case s64: res = arithOp<intl   , op>(lhs, rhs, odims); break;
        case u64: res = arithOp<uintl  , op>(lhs, rhs, odims); break;
        default: TYPE_ERROR(0, otype);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

template<af_op_t op>
static af_err af_arith_real(af_array *out, const af_array lhs, const af_array rhs, const bool batchMode)
{
    try {
        const af_dtype otype = implicit(lhs, rhs);

        ArrayInfo linfo = getInfo(lhs);
        ArrayInfo rinfo = getInfo(rhs);

        dim4 odims = getOutDims(linfo.dims(), rinfo.dims(), batchMode);

        af_array res;
        switch (otype) {
        case f32: res = arithOp<float  , op>(lhs, rhs, odims); break;
        case f64: res = arithOp<double , op>(lhs, rhs, odims); break;
        case s32: res = arithOp<int    , op>(lhs, rhs, odims); break;
        case u32: res = arithOp<uint   , op>(lhs, rhs, odims); break;
        case u8 : res = arithOp<uchar  , op>(lhs, rhs, odims); break;
        case b8 : res = arithOp<char   , op>(lhs, rhs, odims); break;
        case s64: res = arithOp<intl   , op>(lhs, rhs, odims); break;
        case u64: res = arithOp<uintl  , op>(lhs, rhs, odims); break;
        default: TYPE_ERROR(0, otype);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_add(af_array *out, const af_array lhs, const af_array rhs, const bool batchMode)
{
    return af_arith<af_add_t>(out, lhs, rhs, batchMode);
}

af_err af_mul(af_array *out, const af_array lhs, const af_array rhs, const bool batchMode)
{
    return af_arith<af_mul_t>(out, lhs, rhs, batchMode);
}

af_err af_sub(af_array *out, const af_array lhs, const af_array rhs, const bool batchMode)
{
    return af_arith<af_sub_t>(out, lhs, rhs, batchMode);
}

af_err af_div(af_array *out, const af_array lhs, const af_array rhs, const bool batchMode)
{
    return af_arith<af_div_t>(out, lhs, rhs, batchMode);
}

af_err af_maxof(af_array *out, const af_array lhs, const af_array rhs, const bool batchMode)
{
    return af_arith<af_max_t>(out, lhs, rhs, batchMode);
}

af_err af_minof(af_array *out, const af_array lhs, const af_array rhs, const bool batchMode)
{
    return af_arith<af_min_t>(out, lhs, rhs, batchMode);
}

af_err af_rem(af_array *out, const af_array lhs, const af_array rhs, const bool batchMode)
{
    return af_arith_real<af_rem_t>(out, lhs, rhs, batchMode);
}

af_err af_mod(af_array *out, const af_array lhs, const af_array rhs, const bool batchMode)
{
    return af_arith_real<af_mod_t>(out, lhs, rhs, batchMode);
}

af_err af_pow(af_array *out, const af_array lhs, const af_array rhs, const bool batchMode)
{
    try {
        ArrayInfo linfo = getInfo(lhs);
        ArrayInfo rinfo = getInfo(rhs);
        if (linfo.isComplex() || rinfo.isComplex()) {
            AF_ERROR("Powers of Complex numbers not supported", AF_ERR_NOT_SUPPORTED);
        }
    } CATCHALL;

    return af_arith_real<af_pow_t>(out, lhs, rhs, batchMode);
}

af_err af_root(af_array *out, const af_array lhs, const af_array rhs, const bool batchMode)
{
    try {
        ArrayInfo linfo = getInfo(lhs);
        ArrayInfo rinfo = getInfo(rhs);
        if (linfo.isComplex() || rinfo.isComplex()) {
            AF_ERROR("Powers of Complex numbers not supported", AF_ERR_NOT_SUPPORTED);
        }

        af_array one;
        AF_CHECK(af_constant(&one, 1, linfo.ndims(), linfo.dims().get(), linfo.getType()));

        af_array inv_lhs;
        AF_CHECK(af_div(&inv_lhs, one, lhs, batchMode));

        AF_CHECK(af_arith_real<af_pow_t>(out, rhs, inv_lhs, batchMode));

        AF_CHECK(af_release_array(one));
        AF_CHECK(af_release_array(inv_lhs));

    } CATCHALL;

    return AF_SUCCESS;
}

af_err af_atan2(af_array *out, const af_array lhs, const af_array rhs, const bool batchMode)
{
    try {

        const af_dtype type = implicit(lhs, rhs);

        if (type != f32 && type != f64) {
            AF_ERROR("Only floating point arrays are supported for atan2 ",
                     AF_ERR_NOT_SUPPORTED);
        }

        ArrayInfo linfo = getInfo(lhs);
        ArrayInfo rinfo = getInfo(rhs);

        dim4 odims = getOutDims(linfo.dims(), rinfo.dims(), batchMode);

        af_array res;
        switch (type) {
        case f32: res = arithOp<float , af_atan2_t>(lhs, rhs, odims); break;
        case f64: res = arithOp<double, af_atan2_t>(lhs, rhs, odims); break;
        default: TYPE_ERROR(0, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_hypot(af_array *out, const af_array lhs, const af_array rhs, const bool batchMode)
{
    try {

        const af_dtype type = implicit(lhs, rhs);

        if (type != f32 && type != f64) {
            AF_ERROR("Only floating point arrays are supported for hypot ",
                     AF_ERR_NOT_SUPPORTED);
        }

        ArrayInfo linfo = getInfo(lhs);
        ArrayInfo rinfo = getInfo(rhs);

        dim4 odims = getOutDims(linfo.dims(), rinfo.dims(), batchMode);

        af_array res;
        switch (type) {
        case f32: res = arithOp<float , af_hypot_t>(lhs, rhs, odims); break;
        case f64: res = arithOp<double, af_hypot_t>(lhs, rhs, odims); break;
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
    af_array res = getHandle(logicOp<T, op>(castArray<T>(lhs), castArray<T>(rhs), odims));
    return res;
}

template<af_op_t op>
static af_err af_logic(af_array *out, const af_array lhs, const af_array rhs, const bool batchMode)
{
    try {
        const af_dtype type = implicit(lhs, rhs);

        ArrayInfo linfo = getInfo(lhs);
        ArrayInfo rinfo = getInfo(rhs);

        dim4 odims = getOutDims(linfo.dims(), rinfo.dims(), batchMode);

        af_array res;
        switch (type) {
        case f32: res = logicOp<float  , op>(lhs, rhs, odims); break;
        case f64: res = logicOp<double , op>(lhs, rhs, odims); break;
        case c32: res = logicOp<cfloat , op>(lhs, rhs, odims); break;
        case c64: res = logicOp<cdouble, op>(lhs, rhs, odims); break;
        case s32: res = logicOp<int    , op>(lhs, rhs, odims); break;
        case u32: res = logicOp<uint   , op>(lhs, rhs, odims); break;
        case u8 : res = logicOp<uchar  , op>(lhs, rhs, odims); break;
        case b8 : res = logicOp<char   , op>(lhs, rhs, odims); break;
        case s64: res = logicOp<intl   , op>(lhs, rhs, odims); break;
        case u64: res = logicOp<uintl  , op>(lhs, rhs, odims); break;
        default: TYPE_ERROR(0, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_eq(af_array *out, const af_array lhs, const af_array rhs, const bool batchMode)
{
    return af_logic<af_eq_t>(out, lhs, rhs, batchMode);
}

af_err af_neq(af_array *out, const af_array lhs, const af_array rhs, const bool batchMode)
{
    return af_logic<af_neq_t>(out, lhs, rhs, batchMode);
}

af_err af_gt(af_array *out, const af_array lhs, const af_array rhs, const bool batchMode)
{
    return af_logic<af_gt_t>(out, lhs, rhs, batchMode);
}

af_err af_ge(af_array *out, const af_array lhs, const af_array rhs, const bool batchMode)
{
    return af_logic<af_ge_t>(out, lhs, rhs, batchMode);
}

af_err af_lt(af_array *out, const af_array lhs, const af_array rhs, const bool batchMode)
{
    return af_logic<af_lt_t>(out, lhs, rhs, batchMode);
}

af_err af_le(af_array *out, const af_array lhs, const af_array rhs, const bool batchMode)
{
    return af_logic<af_le_t>(out, lhs, rhs, batchMode);
}

af_err af_and(af_array *out, const af_array lhs, const af_array rhs, const bool batchMode)
{
    return af_logic<af_and_t>(out, lhs, rhs, batchMode);
}

af_err af_or(af_array *out, const af_array lhs, const af_array rhs, const bool batchMode)
{
    return af_logic<af_or_t>(out, lhs, rhs, batchMode);
}

template<typename T, af_op_t op>
static inline af_array bitOp(const af_array lhs, const af_array rhs, const dim4 &odims)
{
    af_array res = getHandle(bitOp<T, op>(castArray<T>(lhs), castArray<T>(rhs), odims));
    return res;
}

template<af_op_t op>
static af_err af_bitwise(af_array *out, const af_array lhs, const af_array rhs, const bool batchMode)
{
    try {
        const af_dtype type = implicit(lhs, rhs);

        ArrayInfo linfo = getInfo(lhs);
        ArrayInfo rinfo = getInfo(rhs);

        dim4 odims = getOutDims(linfo.dims(), rinfo.dims(), batchMode);

        af_array res;
        switch (type) {
        case s32: res = bitOp<int    , op>(lhs, rhs, odims); break;
        case u32: res = bitOp<uint   , op>(lhs, rhs, odims); break;
        case u8 : res = bitOp<uchar  , op>(lhs, rhs, odims); break;
        case b8 : res = bitOp<char   , op>(lhs, rhs, odims); break;
        case s64: res = bitOp<intl   , op>(lhs, rhs, odims); break;
        case u64: res = bitOp<uintl  , op>(lhs, rhs, odims); break;
        default: TYPE_ERROR(0, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_bitand(af_array *out, const af_array lhs, const af_array rhs, const bool batchMode)
{
    return af_bitwise<af_bitand_t>(out, lhs, rhs, batchMode);
}

af_err af_bitor(af_array *out, const af_array lhs, const af_array rhs, const bool batchMode)
{
    return af_bitwise<af_bitor_t>(out, lhs, rhs, batchMode);
}

af_err af_bitxor(af_array *out, const af_array lhs, const af_array rhs, const bool batchMode)
{
    return af_bitwise<af_bitxor_t>(out, lhs, rhs, batchMode);
}

af_err af_bitshiftl(af_array *out, const af_array lhs, const af_array rhs, const bool batchMode)
{
    return af_bitwise<af_bitshiftl_t>(out, lhs, rhs, batchMode);
}

af_err af_bitshiftr(af_array *out, const af_array lhs, const af_array rhs, const bool batchMode)
{
    return af_bitwise<af_bitshiftr_t>(out, lhs, rhs, batchMode);
}
