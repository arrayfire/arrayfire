#include <af/array.h>
#include <af/defines.h>
#include <af/arith.h>
#include <ArrayInfo.hpp>
#include <optypes.hpp>
#include <arith.hpp>
#include <logic.hpp>
#include <cast.hpp>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <utility>
#include <map>
using namespace detail;

static af_dtype implicit(const af_array lhs, const af_array rhs)
{
    ArrayInfo lInfo = getInfo(lhs);
    ArrayInfo rInfo = getInfo(rhs);

    if (lInfo.getType() == rInfo.getType()) {
        return lInfo.getType();
    }

    if (lInfo.isComplex() || rInfo.isComplex()) {
        if (lInfo.isDouble() && rInfo.isDouble()) return c64;
        if (lInfo.isDouble() && rInfo.isBool()  ) return c64;
        if (lInfo.isBool()   && rInfo.isDouble()) return c64;
        return c32;
    }

    af_dtype ltype = lInfo.getType();
    af_dtype rtype = lInfo.getType();

    if ((ltype == u32) ||
        (rtype == u32)) return u32;

    if ((ltype == s32) ||
        (rtype == s32)) return s32;

    if ((ltype == u8 ) ||
        (rtype == u8 )) return u8;

    if ((ltype == s8 ) ||
        (rtype == s8 )) return s8;

    if ((ltype == b8 ) &&
        (rtype == b8 )) return b8;

    if (lInfo.isDouble() && rInfo.isDouble()) return f64;
    if (lInfo.isDouble() && rInfo.isBool()  ) return f64;
    if (lInfo.isBool()   && rInfo.isDouble()) return f64;

    return f32;
}


template<typename To>
static af_array cast(const af_array in)
{
    const ArrayInfo info = getInfo(in);
    switch (info.getType()) {
    case f32: return getHandle(*cast<To, float  >(getArray<float  >(in)));
    case f64: return getHandle(*cast<To, double >(getArray<double >(in)));
    case c32: return getHandle(*cast<To, cfloat >(getArray<cfloat >(in)));
    case c64: return getHandle(*cast<To, cdouble>(getArray<cdouble>(in)));
    case s32: return getHandle(*cast<To, int    >(getArray<int    >(in)));
    case u32: return getHandle(*cast<To, uint   >(getArray<uint   >(in)));
    case s8 : return getHandle(*cast<To, char   >(getArray<char   >(in)));
    case u8 : return getHandle(*cast<To, uchar  >(getArray<uchar  >(in)));
    case b8 : return getHandle(*cast<To, uchar  >(getArray<uchar  >(in)));
    default: TYPE_ERROR(1, info.getType());
    }
}

static af_array cast(const af_array in, const af_dtype type)
{
    const ArrayInfo info = getInfo(in);

    if (info.getType() == type) {
        return in;
    }

    switch (type) {
    case f32: return cast<float   >(in);
    case f64: return cast<double  >(in);
    case c32: return cast<cfloat  >(in);
    case c64: return cast<cdouble >(in);
    case s32: return cast<int     >(in);
    case u32: return cast<uint    >(in);
    case s8 : return cast<char    >(in);
    case u8 : return cast<uchar   >(in);
    case b8 : return cast<uchar   >(in);
    default: TYPE_ERROR(2, type);
    }
}

af_err af_cast(af_array *out, const af_array in, const af_dtype type)
{
    try {
        af_array res = cast(in, type);
        std::swap(*out, res);
    }
    CATCHALL;

    return AF_SUCCESS;
}

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
        const af_array right = cast(lhs, type);

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
