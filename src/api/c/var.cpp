/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/statistics.h>
#include <af/defines.h>
#include <backend.hpp>
#include <reduce.hpp>
#include <handle.hpp>
#include <arith.hpp>
#include <math.hpp>
#include <cast.hpp>

using namespace detail;

template<typename inType, typename outType>
static outType var(const af_array& in, bool isbiased)
{
    const Array<outType> *input = cast<outType>(getArray<inType>(in));

    outType mean = division(reduce_all<af_add_t, outType, outType>(*input),
        input->elements());
    Array<outType> *meanArr = createValueArray<outType>(input->dims(), mean);
    Array<outType> *tmp = detail::arithOp<outType, af_sub_t>(*input, *meanArr,
        input->dims());
    tmp = detail::arithOp<outType, af_mul_t>(*tmp, *tmp, tmp->dims());

    outType result = division(reduce_all<af_add_t, outType, outType>(*tmp),
        isbiased ? input->elements() : input->elements() - 1);

    destroyArray<outType>(*meanArr);
    destroyArray<outType>(*tmp);

    return result;
}

template<typename inType, typename outType>
static outType var(const af_array& in, const af_array weights)
{
    return scalar<outType>(0);
}

af_err af_var_all(double *realVal, double *imagVal, const af_array in,
    bool isbiased)
{
    try {
        ArrayInfo info = getInfo(in);
        af_dtype type = info.getType();
        switch(type) {
            case f64: *realVal = var<double, double>(in, isbiased); break;
            case f32: *realVal = var<float , float >(in, isbiased); break;
            case s32: *realVal = var<int   , int   >(in, isbiased); break;
            case u32: *realVal = var<uint  , uint  >(in, isbiased); break;
            case  u8: *realVal = var<uchar , uint  >(in, isbiased); break;
            case  b8: *realVal = var<char  , int   >(in, isbiased); break;
            case c32: {
                cfloat tmp = var<cfloat,cfloat>(in, isbiased);
                *realVal = real(tmp);
                *imagVal = imag(tmp);
                } break;
            case c64: {
                cdouble tmp = var<cdouble,cdouble>(in, isbiased);
                *realVal = real(tmp);
                *imagVal = imag(tmp);
                } break;
            default : TYPE_ERROR(1, type);
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_var_all_weighted(double *realVal, double *imagVal, const af_array in,
    const af_array weights)
{
    try {
        ArrayInfo info = getInfo(in);
        af_dtype type = info.getType();
        switch(type) {
            case f64: *realVal = var<double, double>(in, weights); break;
            case f32: *realVal = var<float , float >(in, weights); break;
            case s32: *realVal = var<int   , int   >(in, weights); break;
            case u32: *realVal = var<uint  , uint  >(in, weights); break;
            case  u8: *realVal = var<uchar , uint  >(in, weights); break;
            case  b8: *realVal = var<char  , int   >(in, weights); break;
            case c32: {
                cfloat tmp = var<cfloat,cfloat>(in, weights);
                *realVal = real(tmp);
                *imagVal = imag(tmp);
                } break;
            case c64: {
                cdouble tmp = var<cdouble,cdouble>(in, weights);
                *realVal = real(tmp);
                *imagVal = imag(tmp);
                } break;
            default : TYPE_ERROR(1, type);
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}
