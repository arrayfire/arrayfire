#include <af/array.h>
#include <af/defines.h>
#include <af/arith.h>
#include <ArrayInfo.hpp>
#include <optypes.hpp>
#include <implicit.hpp>

#include <arith.hpp>
#include <logic.hpp>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>

using namespace detail;

template<typename T, af_op_t op>
static inline af_array arithOp(const af_array lhs, const af_array rhs)
{
    return getHandle(*arithOp<T, op>(getArray<T>(lhs), getArray<T>(rhs)));
}

template<af_op_t op>
static af_err af_arith(af_array *out, const af_array lhs, const af_array rhs)
{
    try {
        const af_dtype otype = implicit(lhs, rhs);
        const af_array left  = cast(lhs, otype);
        const af_array right = cast(rhs, otype);

        af_array res;
        switch (otype) {
        case f32: res = arithOp<float  , op>(left, right); break;
        case f64: res = arithOp<double , op>(left, right); break;
        case c32: res = arithOp<cfloat , op>(left, right); break;
        case c64: res = arithOp<cdouble, op>(left, right); break;
        case s32: res = arithOp<int    , op>(left, right); break;
        case u32: res = arithOp<uint   , op>(left, right); break;
        case s8 : res = arithOp<char   , op>(left, right); break;
        case u8 : res = arithOp<uchar  , op>(left, right); break;
        case b8 : res = arithOp<uchar  , op>(left, right); break;
        default: TYPE_ERROR(0, otype);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_add(af_array *out, const af_array lhs, const af_array rhs)
{
    return af_arith<af_add_t>(out, lhs, rhs);
}

af_err af_mul(af_array *out, const af_array lhs, const af_array rhs)
{
    return af_arith<af_mul_t>(out, lhs, rhs);
}

af_err af_sub(af_array *out, const af_array lhs, const af_array rhs)
{
    return af_arith<af_sub_t>(out, lhs, rhs);
}

af_err af_div(af_array *out, const af_array lhs, const af_array rhs)
{
    return af_arith<af_div_t>(out, lhs, rhs);
}

template<typename T, af_op_t op>
static inline af_array logicOp(const af_array lhs, const af_array rhs)
{
    return getHandle(*logicOp<T, op>(getArray<T>(lhs), getArray<T>(rhs)));
}

template<af_op_t op>
static af_err af_logic(af_array *out, const af_array lhs, const af_array rhs)
{
    try {
        const af_dtype type = implicit(lhs, rhs);

        const af_array left  = cast(lhs, type);
        const af_array right = cast(rhs, type);

        af_array res;
        switch (type) {
        case f32: res = logicOp<float  , op>(left, right); break;
        case f64: res = logicOp<double , op>(left, right); break;
        case c32: res = logicOp<cfloat , op>(left, right); break;
        case c64: res = logicOp<cdouble, op>(left, right); break;
        case s32: res = logicOp<int    , op>(left, right); break;
        case u32: res = logicOp<uint   , op>(left, right); break;
        case s8 : res = logicOp<char   , op>(left, right); break;
        case u8 : res = logicOp<uchar  , op>(left, right); break;
        case b8 : res = logicOp<uchar  , op>(left, right); break;
        default: TYPE_ERROR(0, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_eq(af_array *out, const af_array lhs, const af_array rhs)
{
    return af_logic<af_eq_t>(out, lhs, rhs);
}

af_err af_neq(af_array *out, const af_array lhs, const af_array rhs)
{
    return af_logic<af_neq_t>(out, lhs, rhs);
}

af_err af_gt(af_array *out, const af_array lhs, const af_array rhs)
{
    return af_logic<af_gt_t>(out, lhs, rhs);
}

af_err af_ge(af_array *out, const af_array lhs, const af_array rhs)
{
    return af_logic<af_ge_t>(out, lhs, rhs);
}

af_err af_lt(af_array *out, const af_array lhs, const af_array rhs)
{
    return af_logic<af_lt_t>(out, lhs, rhs);
}

af_err af_le(af_array *out, const af_array lhs, const af_array rhs)
{
    return af_logic<af_le_t>(out, lhs, rhs);
}
