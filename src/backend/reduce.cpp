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
#include <af/index.h>
#include <af/algorithm.h>
#include <err_common.hpp>
#include <handle.hpp>
#include <ops.hpp>
#include <backend.hpp>
#include <reduce.hpp>

using af::dim4;
using namespace detail;

template<af_op_t op, typename Ti, typename To>
static inline af_array reduce(const af_array in, const int dim)
{
    return getHandle(*reduce<op,Ti,To>(getArray<Ti>(in), dim));
}

template<af_op_t op, typename To>
static af_err reduce_type(af_array *out, const af_array in, const int dim)
{
    try {

        ARG_ASSERT(2, dim >= 0);
        ARG_ASSERT(2, dim <  4);

        const ArrayInfo in_info = getInfo(in);

        if (dim >= (int)in_info.ndims()) {
            *out = weakCopy(in);
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
            *out = weakCopy(in);
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
            *out = weakCopy(in);
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

af_err af_count(af_array *out, const af_array in, const int dim)
{
    return reduce_type<af_notzero_t, uint>(out, in, dim);
}

af_err af_alltrue(af_array *out, const af_array in, const int dim)
{
    return reduce_type<af_and_t, uchar>(out, in, dim);
}

af_err af_anytrue(af_array *out, const af_array in, const int dim)
{
    return reduce_type<af_or_t, uchar>(out, in, dim);
}
