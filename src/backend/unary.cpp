#include <af/array.h>
#include <af/defines.h>
#include <af/arith.h>
#include <ArrayInfo.hpp>
#include <optypes.hpp>
#include <implicit.hpp>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <unary.hpp>
#include <TNJ/UnaryNode.hpp>

using namespace detail;

template<typename T, af_op_t op>
static inline af_array unaryOp(const af_array in)
{
    return getHandle(*unaryOp<T, op>(getArray<T>(in)));
}

template<af_op_t op>
static af_err af_unary(af_array *out, const af_array in)
{
    try {

        ArrayInfo in_info = getInfo(in);
        ARG_ASSERT(1, in_info.isReal());
        ARG_ASSERT(1, !in_info.isBool());

        af_dtype in_type = in_info.getType();
        af_array res;

        switch (in_type) {
        case f32 : res = unaryOp<float  , op>(in); break;
        case f64 : res = unaryOp<double , op>(in); break;
        case s32 : res = unaryOp<int    , op>(in); break;
        case u32 : res = unaryOp<uint   , op>(in); break;
        case s8  : res = unaryOp<char   , op>(in); break;
        case u8  : res = unaryOp<uchar  , op>(in); break;
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
