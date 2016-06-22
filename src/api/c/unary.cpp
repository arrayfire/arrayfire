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
#include <complex.hpp>
#include <logic.hpp>
#include <cast.hpp>
#include <arith.hpp>

using namespace detail;

template<typename T, af_op_t op>
static inline af_array unaryOp(const af_array in)
{
    af_array res = getHandle(unaryOp<T, op>(castArray<T>(in)));
    return res;
}

template<typename Tc, typename Tr, af_op_t op>
struct unaryOpCplx;

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

        switch (type) {
        case f32 : res = unaryOp<float  , op>(in); break;
        case f64 : res = unaryOp<double , op>(in); break;
        default:
            TYPE_ERROR(1, in_type); break;
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

template<af_op_t op>
static af_err af_unary_complex(af_array *out, const af_array in)
{
    try {
        ArrayInfo in_info = getInfo(in);

        af_dtype in_type = in_info.getType();
        af_array res;

        // Convert all inputs to floats / doubles
        af_dtype type = implicit(in_type, f32);

        switch (type) {
        case f32 : res = unaryOp<float  , op>(in); break;
        case f64 : res = unaryOp<double , op>(in); break;
        case c32 : res = unaryOpCplx<cfloat , float , op>()(in); break;
        case c64 : res = unaryOpCplx<cdouble, double, op>()(in); break;
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

#define UNARY_COMPLEX(fn)                               \
    af_err af_##fn(af_array *out, const af_array in)    \
    {                                                   \
        return af_unary_complex<af_##fn##_t>(out, in);  \
    }

UNARY(asin)
UNARY(acos)
UNARY(atan)

UNARY(sinh)
UNARY(cosh)
UNARY(tanh)

UNARY(asinh)
UNARY(acosh)
UNARY(atanh)

UNARY(trunc)
UNARY(sign)
UNARY(round)
UNARY(floor)
UNARY(ceil)

UNARY(sigmoid)
UNARY(expm1)
UNARY(erf)
UNARY(erfc)

UNARY(log10)
UNARY(log1p)
UNARY(log2)

UNARY(cbrt)

UNARY(tgamma)
UNARY(lgamma)

UNARY_COMPLEX(cos)
UNARY_COMPLEX(exp)
UNARY_COMPLEX(log)
UNARY_COMPLEX(sin)
UNARY_COMPLEX(sqrt)
UNARY_COMPLEX(tan)

template<typename Tc, typename Tr>
struct unaryOpCplx<Tc, Tr, af_exp_t>
{
    af_array operator()(const af_array a)
    {
        Array<Tc> In = getArray<Tc>(a);
        Array<Tr> Real = real<Tr, Tc>(In);
        Array<Tr> Imag = imag<Tr, Tc>(In);

        Array<Tr> ExpReal = unaryOp<Tr, af_exp_t>(Real);
        Array<Tr> CosImag = unaryOp<Tr, af_cos_t>(Imag);
        Array<Tr> SinImag = unaryOp<Tr, af_sin_t>(Imag);

        Array<Tc> Unit  = cplx<Tc, Tr>(CosImag, SinImag, CosImag.dims());
        Array<Tc> Scale = cast<Tc, Tr>(ExpReal);

        Array<Tc> Result = arithOp<Tc, af_mul_t>(Scale, Unit, Scale.dims());

        return getHandle(Result);
    }
};

template<typename Tc, typename Tr>
struct unaryOpCplx<Tc, Tr, af_log_t>
{
    af_array operator()(const af_array in)
    {
        // convert cartesian to polar
        Array<Tc> z = getArray<Tc>(in);
        Array<Tr> a = real<Tr, Tc>(z);
        Array<Tr> b = imag<Tr, Tc>(z);
        Array<Tr> r = arithOp<Tr, af_atan2_t>(b, a, b.dims());
        Array<Tr> phi = abs<Tr>(z);

        // compute log
        Array<Tr> a_out = unaryOp<Tr, af_log_t>(r);
        Array<Tr> b_out = phi;
        Array<Tc> z_out  = cplx<Tc, Tr>(a_out, b_out, a_out.dims());

        return getHandle(z_out);
    }
};

template<typename Tc, typename Tr>
struct unaryOpCplx<Tc, Tr, af_sin_t>
{
    af_array operator()(const af_array in)
    {
        Array<Tc> z = getArray<Tc>(in);
        Array<Tr> a = real<Tr, Tc>(z);
        Array<Tr> b = imag<Tr, Tc>(z);

        // compute sin
        Array<Tr> sin_a = unaryOp<Tr, af_sin_t>(a);
        Array<Tr> cos_a = unaryOp<Tr, af_cos_t>(a);
        Array<Tr> sinh_b = unaryOp<Tr, af_sinh_t>(b);
        Array<Tr> cosh_b = unaryOp<Tr, af_cosh_t>(b);
        Array<Tr> a_out = arithOp<Tr, af_mul_t>(sin_a, cosh_b, sin_a.dims());
        Array<Tr> b_out = arithOp<Tr, af_mul_t>(cos_a, sinh_b, cos_a.dims());

        Array<Tc> z_out  = cplx<Tc, Tr>(a_out, b_out, a_out.dims());

        return getHandle(z_out);
    }
};

template<typename Tc, typename Tr>
struct unaryOpCplx<Tc, Tr, af_cos_t>
{
    af_array operator()(const af_array in)
    {
        Array<Tc> z = getArray<Tc>(in);
        Array<Tr> a = real<Tr, Tc>(z);
        Array<Tr> b = imag<Tr, Tc>(z);

        // compute cos
        Array<Tr> sin_a = unaryOp<Tr, af_sin_t>(a);
        Array<Tr> cos_a = unaryOp<Tr, af_cos_t>(a);
        Array<Tr> sinh_b = unaryOp<Tr, af_sinh_t>(b);
        Array<Tr> cosh_b = unaryOp<Tr, af_cosh_t>(b);
        Array<Tr> a_out = arithOp<Tr, af_mul_t>(cos_a, cosh_b, sin_a.dims());
        Array<Tr> neg_one = createValueArray<Tr>(a_out.dims(), -1);
        Array<Tr> b_out_neg = arithOp<Tr, af_mul_t>(sin_a, sinh_b, cos_a.dims());
        Array<Tr> b_out = arithOp<Tr, af_mul_t>(b_out_neg, b_out_neg, b_out_neg.dims());

        Array<Tc> z_out  = cplx<Tc, Tr>(a_out, b_out, a_out.dims());

        return getHandle(z_out);
    }
};

template<typename Tc, typename Tr>
struct unaryOpCplx<Tc, Tr, af_tan_t>
{
    af_array operator()(const af_array in)
    {
        Array<Tc> sin_z = getArray<Tc>(unaryOpCplx<Tc, Tr, af_sin_t>()(in));
        Array<Tc> cos_z = getArray<Tc>(unaryOpCplx<Tc, Tr, af_cos_t>()(in));
        Array<Tc> tan_z = arithOp<Tc, af_div_t>(sin_z, cos_z, sin_z.dims());
        return getHandle(tan_z);
    }
};

template<typename Tc, typename Tr>
struct unaryOpCplx<Tc, Tr, af_sqrt_t>
{
    af_array operator()(const af_array in)
    {
        // convert cartesian to polar
        Array<Tc> z = getArray<Tc>(in);
        Array<Tr> a = real<Tr, Tc>(z);
        Array<Tr> b = imag<Tr, Tc>(z);
        Array<Tr> r = arithOp<Tr, af_atan2_t>(b, a, b.dims());
        Array<Tr> phi = abs<Tr>(z);

        // compute sqrt
        Array<Tr> two = createValueArray<Tr>(phi.dims(), 2.0);
        Array<Tr> r_out = unaryOp<Tr, af_sqrt_t>(r);
        Array<Tr> phi_out = arithOp<Tr, af_div_t>(phi, two, phi.dims());

        // convert polar to cartesian
        Array<Tr> a_out_unit = unaryOp<Tr, af_cos_t>(phi_out);
        Array<Tr> b_out_unit = unaryOp<Tr, af_sin_t>(phi_out);
        Array<Tr> a_out = arithOp<Tr, af_mul_t>(r_out, a_out_unit, r_out.dims());
        Array<Tr> b_out = arithOp<Tr, af_mul_t>(r_out, b_out_unit, r_out.dims());
        Array<Tc> z_out  = cplx<Tc, Tr>(a_out, b_out, a_out.dims());

        return getHandle(z_out);
    }
};

af_err af_not(af_array *out, const af_array in)
{
    try {

        af_array tmp;
        ArrayInfo in_info = getInfo(in);

        AF_CHECK(af_constant(&tmp, 0,
                             in_info.ndims(),
                             in_info.dims().get(), in_info.getType()));

        AF_CHECK(af_eq(out, in, tmp, false));

        AF_CHECK(af_release_array(tmp));
    } CATCHALL;

    return AF_SUCCESS;
}

af_err af_arg(af_array *out, const af_array in)
{
    try {

        ArrayInfo in_info = getInfo(in);

        if (!in_info.isComplex()) {
            return af_constant(out, 0,
                               in_info.ndims(),
                               in_info.dims().get(), in_info.getType());
        }

        af_array real;
        af_array imag;

        AF_CHECK(af_real(&real, in));
        AF_CHECK(af_imag(&imag, in));

        AF_CHECK(af_atan2(out, imag, real, false));

        AF_CHECK(af_release_array(real));
        AF_CHECK(af_release_array(imag));
    } CATCHALL;

    return AF_SUCCESS;
}

af_err af_pow2(af_array *out, const af_array in)
{
    try {

        af_array two;
        ArrayInfo in_info = getInfo(in);

        AF_CHECK(af_constant(&two, 2,
                             in_info.ndims(),
                             in_info.dims().get(), in_info.getType()));

        AF_CHECK(af_pow(out, two, in, false));

        AF_CHECK(af_release_array(two));
    } CATCHALL;

    return AF_SUCCESS;
}

af_err af_factorial(af_array *out, const af_array in)
{
    try {

        af_array one;
        ArrayInfo in_info = getInfo(in);

        AF_CHECK(af_constant(&one, 1,
                             in_info.ndims(),
                             in_info.dims().get(), in_info.getType()));

        af_array inp1;
        AF_CHECK(af_add(&inp1, one, in, false));

        AF_CHECK(af_tgamma(out, inp1));

        AF_CHECK(af_release_array(one));
        AF_CHECK(af_release_array(inp1));
    } CATCHALL;

    return AF_SUCCESS;
}

template<typename T, af_op_t op>
static inline af_array checkOp(const af_array in)
{
    af_array res = getHandle(checkOp<T, op>(castArray<T>(in)));
    return res;
}

template<af_op_t op>
struct cplxLogicOp
{
    af_array operator()(Array<char> resR, Array<char> resI, dim4 dims)
    {
        return getHandle(logicOp<char, af_or_t>(resR, resI, dims));
    }
};

template <>
struct cplxLogicOp<af_iszero_t>
{
    af_array operator()(Array<char> resR, Array<char> resI, dim4 dims)
    {
        return getHandle(logicOp<char, af_and_t>(resR, resI, dims));
    }
};

template<typename T, typename BT, af_op_t op>
static inline af_array checkOpCplx(const af_array in)
{
    Array<BT> R = real<BT, T>(getArray<T>(in));
    Array<BT> I = imag<BT, T>(getArray<T>(in));

    Array<char> resR = checkOp<BT, op>(R);
    Array<char> resI = checkOp<BT, op>(I);

    ArrayInfo in_info = getInfo(in);
    dim4 dims = in_info.dims();
    cplxLogicOp<op> cplxLogic;
    af_array res = cplxLogic(resR, resI, dims);

    return res;
}

template<af_op_t op>
static af_err af_check(af_array *out, const af_array in)
{
    try {

        ArrayInfo in_info = getInfo(in);

        af_dtype in_type = in_info.getType();
        af_array res;

        // Convert all inputs to floats / doubles / complex
        af_dtype type = implicit(in_type, f32);

        switch (type) {
        case f32 : res = checkOp<float  , op>(in); break;
        case f64 : res = checkOp<double , op>(in); break;
        case c32 : res = checkOpCplx<cfloat , float , op>(in); break;
        case c64 : res = checkOpCplx<cdouble, double, op>(in); break;
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
