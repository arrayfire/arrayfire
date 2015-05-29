/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <complex>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/algorithm.h>
#include <err_common.hpp>
#include <handle.hpp>
#include <ops.hpp>
#include <backend.hpp>
#include <reduce.hpp>
#include <ireduce.hpp>
#include <math.hpp>

using af::dim4;
using namespace detail;

template<af_op_t op, typename Ti, typename To>
static inline af_array reduce(const af_array in, const int dim)
{
    return getHandle(reduce<op,Ti,To>(getArray<Ti>(in), dim));
}

template<af_op_t op, typename To>
static af_err reduce_type(af_array *out, const af_array in, const int dim)
{
    try {

        ARG_ASSERT(2, dim >= 0);
        ARG_ASSERT(2, dim <  4);

        const ArrayInfo in_info = getInfo(in);

        if (dim >= (int)in_info.ndims()) {
            *out = retain(in);
            return AF_SUCCESS;
        }

        af_dtype type = in_info.getType();
        af_array res;

        switch(type) {
        case f32:  res = reduce<op, float  , To>(in, dim); break;
        case f64:  res = reduce<op, double , To>(in, dim); break;
        case c32:  res = reduce<op, cfloat , To>(in, dim); break;
        case c64:  res = reduce<op, cdouble, To>(in, dim); break;
        case u32:  res = reduce<op, uint   , To>(in, dim); break;
        case s32:  res = reduce<op, int    , To>(in, dim); break;
        case b8:   res = reduce<op, char   , To>(in, dim); break;
        case u8:   res = reduce<op, uchar  , To>(in, dim); break;
        default:   TYPE_ERROR(1, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<af_op_t op>
static af_err reduce_common(af_array *out, const af_array in, const int dim)
{
    try {

        ARG_ASSERT(2, dim >= 0);
        ARG_ASSERT(2, dim <  4);

        const ArrayInfo in_info = getInfo(in);

        if (dim >= (int)in_info.ndims()) {
            *out = retain(in);
            return AF_SUCCESS;
        }

        af_dtype type = in_info.getType();
        af_array res;

        switch(type) {
        case f32:  res = reduce<op, float  , float  >(in, dim); break;
        case f64:  res = reduce<op, double , double >(in, dim); break;
        case c32:  res = reduce<op, cfloat , cfloat >(in, dim); break;
        case c64:  res = reduce<op, cdouble, cdouble>(in, dim); break;
        case u32:  res = reduce<op, uint   , uint   >(in, dim); break;
        case s32:  res = reduce<op, int    , int    >(in, dim); break;
        case b8:   res = reduce<op, char   , char   >(in, dim); break;
        case u8:   res = reduce<op, uchar  , uchar  >(in, dim); break;
        default:   TYPE_ERROR(1, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<af_op_t op>
static af_err reduce_promote(af_array *out, const af_array in, const int dim)
{
    try {

        ARG_ASSERT(2, dim >= 0);
        ARG_ASSERT(2, dim <  4);

        const ArrayInfo in_info = getInfo(in);

        if (dim >= (int)in_info.ndims()) {
            *out = retain(in);
            return AF_SUCCESS;
        }

        af_dtype type = in_info.getType();
        af_array res;

        switch(type) {
        case f32:  res = reduce<op, float  , float  >(in, dim); break;
        case f64:  res = reduce<op, double , double >(in, dim); break;
        case c32:  res = reduce<op, cfloat , cfloat >(in, dim); break;
        case c64:  res = reduce<op, cdouble, cdouble>(in, dim); break;
        case u32:  res = reduce<op, uint   , uint   >(in, dim); break;
        case s32:  res = reduce<op, int    , int    >(in, dim); break;
        case u8:   res = reduce<op, uchar  , uint   >(in, dim); break;
            // Make sure you are adding only "1" for every non zero value, even if op == af_add_t
        case b8:   res = reduce<af_notzero_t, char  , uint   >(in, dim); break;
        default:   TYPE_ERROR(1, type);
        }
        std::swap(*out, res);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_min(af_array *out, const af_array in, const int dim)
{
    return reduce_common<af_min_t>(out, in, dim);
}

af_err af_max(af_array *out, const af_array in, const int dim)
{
    return reduce_common<af_max_t>(out, in, dim);
}

af_err af_sum(af_array *out, const af_array in, const int dim)
{
    return reduce_promote<af_add_t>(out, in, dim);
}

af_err af_product(af_array *out, const af_array in, const int dim)
{
    return reduce_promote<af_mul_t>(out, in, dim);
}

af_err af_count(af_array *out, const af_array in, const int dim)
{
    return reduce_type<af_notzero_t, uint>(out, in, dim);
}

af_err af_all_true(af_array *out, const af_array in, const int dim)
{
    return reduce_type<af_and_t, char>(out, in, dim);
}

af_err af_any_true(af_array *out, const af_array in, const int dim)
{
    return reduce_type<af_or_t, char>(out, in, dim);
}

template<af_op_t op, typename Ti, typename To>
static inline To reduce_all(const af_array in)
{
    return reduce_all<op,Ti,To>(getArray<Ti>(in));
}

template<af_op_t op, typename To>
static af_err reduce_all_type(double *real, double *imag, const af_array in)
{
    try {

        const ArrayInfo in_info = getInfo(in);
        af_dtype type = in_info.getType();

        ARG_ASSERT(0, real != NULL);
        *real = 0;
        if (!imag) *imag = 0;

        switch(type) {
        case f32:  *real = (double)reduce_all<op, float  , To>(in); break;
        case f64:  *real = (double)reduce_all<op, double , To>(in); break;
        case c32:  *real = (double)reduce_all<op, cfloat , To>(in); break;
        case c64:  *real = (double)reduce_all<op, cdouble, To>(in); break;
        case u32:  *real = (double)reduce_all<op, uint   , To>(in); break;
        case s32:  *real = (double)reduce_all<op, int    , To>(in); break;
        case b8:   *real = (double)reduce_all<op, char   , To>(in); break;
        case u8:   *real = (double)reduce_all<op, uchar  , To>(in); break;
        default:   TYPE_ERROR(1, type);
        }

    }
    CATCHALL;

    return AF_SUCCESS;
}

template<af_op_t op>
static af_err reduce_all_common(double *real_val, double *imag_val, const af_array in)
{
    try {

        const ArrayInfo in_info = getInfo(in);
        af_dtype type = in_info.getType();

        ARG_ASSERT(0, real_val != NULL);
        *real_val = 0;
        if (!imag_val) *imag_val = 0;

        cfloat  cfval;
        cdouble cdval;

        switch(type) {
        case f32:  *real_val = (double)reduce_all<op, float  , float  >(in); break;
        case f64:  *real_val = (double)reduce_all<op, double , double >(in); break;
        case u32:  *real_val = (double)reduce_all<op, uint   , uint   >(in); break;
        case s32:  *real_val = (double)reduce_all<op, int    , int    >(in); break;
        case b8:   *real_val = (double)reduce_all<op, char   , char   >(in); break;
        case u8:   *real_val = (double)reduce_all<op, uchar  , uchar  >(in); break;

        case c32:
            cfval = reduce_all<op, cfloat, cfloat>(in);
            ARG_ASSERT(1, imag_val != NULL);
            *real_val = real(cfval);
            *imag_val = imag(cfval);
            break;

        case c64:
            cdval = reduce_all<op, cdouble, cdouble>(in);
            ARG_ASSERT(1, imag_val != NULL);
            *real_val = real(cdval);
            *imag_val = imag(cdval);
            break;

        default:   TYPE_ERROR(1, type);
        }

    }
    CATCHALL;

    return AF_SUCCESS;
}

template<af_op_t op>
static af_err reduce_all_promote(double *real_val, double *imag_val, const af_array in)
{
    try {

        const ArrayInfo in_info = getInfo(in);
        af_dtype type = in_info.getType();

        ARG_ASSERT(0, real_val != NULL);
        *real_val = 0;
        if (!imag_val) *imag_val = 0;

        cfloat  cfval;
        cdouble cdval;

        switch(type) {
        case f32: *real_val = (double)reduce_all<op, float  , float  >(in); break;
        case f64: *real_val = (double)reduce_all<op, double , double >(in); break;
        case u32: *real_val = (double)reduce_all<op, uint   , uint   >(in); break;
        case s32: *real_val = (double)reduce_all<op, int    , int    >(in); break;
        case u8:  *real_val = (double)reduce_all<op, uchar  , uint   >(in); break;
            // Make sure you are adding only "1" for every non zero value, even if op == af_add_t
        case b8:  *real_val = (double)reduce_all<af_notzero_t, char  , uint   >(in); break;

        case c32:
            cfval = reduce_all<op, cfloat, cfloat>(in);
            ARG_ASSERT(1, imag_val != NULL);
            *real_val = real(cfval);
            *imag_val = imag(cfval);
            break;

        case c64:
            cdval = reduce_all<op, cdouble, cdouble>(in);
            ARG_ASSERT(1, imag_val != NULL);
            *real_val = real(cdval);
            *imag_val = imag(cdval);
            break;

        default:   TYPE_ERROR(1, type);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_min_all(double *real, double *imag, const af_array in)
{
    return reduce_all_common<af_min_t>(real, imag, in);
}

af_err af_max_all(double *real, double *imag, const af_array in)
{
    return reduce_all_common<af_max_t>(real, imag, in);
}

af_err af_sum_all(double *real, double *imag, const af_array in)
{
    return reduce_all_promote<af_add_t>(real, imag, in);
}

af_err af_product_all(double *real, double *imag, const af_array in)
{
    return reduce_all_promote<af_mul_t>(real, imag, in);
}

af_err af_count_all(double *real, double *imag, const af_array in)
{
    return reduce_all_type<af_notzero_t, uint>(real, imag, in);
}

af_err af_all_true_all(double *real, double *imag, const af_array in)
{
    return reduce_all_type<af_and_t, char>(real, imag, in);
}

af_err af_any_true_all(double *real, double *imag, const af_array in)
{
    return reduce_all_type<af_or_t , char>(real, imag, in);
}

template<af_op_t op, typename T>
static inline void ireduce(af_array *res, af_array *loc,
                           const af_array in, const int dim)
{
    const Array<T> In = getArray<T>(in);
    dim4 odims = In.dims();
    odims[dim] = 1;

    Array<T> Res = createEmptyArray<T>(odims);
    Array<uint> Loc = createEmptyArray<uint>(odims);
    ireduce<op, T>(Res, Loc, In, dim);

    *res = getHandle(Res);
    *loc = getHandle(Loc);
}

template<af_op_t op>
static af_err ireduce_common(af_array *val, af_array *idx, const af_array in, const int dim)
{
    try {

        ARG_ASSERT(2, dim >= 0);
        ARG_ASSERT(2, dim <  4);

        const ArrayInfo in_info = getInfo(in);

        if (dim >= (int)in_info.ndims()) {
            *val = retain(in);
            return AF_SUCCESS;
        }

        af_dtype type = in_info.getType();
        af_array res, loc;

        switch(type) {
        case f32:  ireduce<op, float  >(&res, &loc, in, dim); break;
        case f64:  ireduce<op, double >(&res, &loc, in, dim); break;
        case c32:  ireduce<op, cfloat >(&res, &loc, in, dim); break;
        case c64:  ireduce<op, cdouble>(&res, &loc, in, dim); break;
        case u32:  ireduce<op, uint   >(&res, &loc, in, dim); break;
        case s32:  ireduce<op, int    >(&res, &loc, in, dim); break;
        case b8:   ireduce<op, char   >(&res, &loc, in, dim); break;
        case u8:   ireduce<op, uchar  >(&res, &loc, in, dim); break;
        default:   TYPE_ERROR(1, type);
        }

        std::swap(*val, res);
        std::swap(*idx, loc);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_imin(af_array *val, af_array *idx, const af_array in, const int dim)
{
    return ireduce_common<af_min_t>(val, idx, in, dim);
}

af_err af_imax(af_array *val, af_array *idx, const af_array in, const int dim)
{
    return ireduce_common<af_max_t>(val, idx, in, dim);
}

template<af_op_t op, typename T>
static inline T ireduce_all(unsigned *loc, const af_array in)
{
    return ireduce_all<op, T>(loc, getArray<T>(in));
}

template<af_op_t op>
static af_err ireduce_all_common(double *real_val, double *imag_val,
                                 unsigned *loc, const af_array in)
{
    try {

        const ArrayInfo in_info = getInfo(in);
        af_dtype type = in_info.getType();

        ARG_ASSERT(0, real_val != NULL);
        *real_val = 0;
        if (!imag_val) *imag_val = 0;

        cfloat  cfval;
        cdouble cdval;

        switch(type) {
        case f32:  *real_val = (double)ireduce_all<op, float >(loc, in); break;
        case f64:  *real_val = (double)ireduce_all<op, double>(loc, in); break;
        case u32:  *real_val = (double)ireduce_all<op, uint  >(loc, in); break;
        case s32:  *real_val = (double)ireduce_all<op, int   >(loc, in); break;
        case b8:   *real_val = (double)ireduce_all<op, char  >(loc, in); break;
        case u8:   *real_val = (double)ireduce_all<op, uchar >(loc, in); break;

        case c32:
            cfval = ireduce_all<op, cfloat>(loc, in);
            ARG_ASSERT(1, imag_val != NULL);
            *real_val = real(cfval);
            *imag_val = imag(cfval);
            break;

        case c64:
            cdval = ireduce_all<op, cdouble>(loc, in);
            ARG_ASSERT(1, imag_val != NULL);
            *real_val = real(cdval);
            *imag_val = imag(cdval);
            break;

        default:   TYPE_ERROR(1, type);
        }

    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_imin_all(double *real, double *imag, unsigned *idx, const af_array in)
{
    return ireduce_all_common<af_min_t>(real, imag, idx, in);
}

af_err af_imax_all(double *real, double *imag, unsigned *idx, const af_array in)
{
    return ireduce_all_common<af_max_t>(real, imag, idx, in);
}
