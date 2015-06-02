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
#include <err_common.hpp>
#include <backend.hpp>
#include <handle.hpp>
#include <reduce.hpp>
#include <arith.hpp>
#include <math.hpp>
#include <cast.hpp>
#include <tile.hpp>

#include "stats.h"

using namespace detail;

template<typename inType, typename outType>
static outType varAll(const af_array& in, const bool isbiased)
{
    Array<outType> input = cast<outType>(getArray<inType>(in));

    Array<outType> meanCnst= createValueArray<outType>(input.dims(), mean<outType>(input));

    Array<outType> diff    = arithOp<outType, af_sub_t>(input, meanCnst, input.dims());

    Array<outType> diffSq  = arithOp<outType, af_mul_t>(diff, diff, diff.dims());

    outType result = division(reduce_all<af_add_t, outType, outType>(diffSq),
        isbiased ? input.elements() : input.elements() - 1);

    return result;
}

template<typename inType, typename outType>
static outType varAll(const af_array& in, const af_array weights)
{
    typedef typename baseOutType<outType>::type bType;

    Array<outType> input = cast<outType>(getArray<inType>(in));
    Array<outType> wts   = cast<outType>(getArray<bType>(weights));

    bType wtsSum    = reduce_all<af_add_t, bType, bType>(getArray<bType>(weights));
    outType wtdMean = mean<outType, bType>(input, getArray<bType>(weights));

    Array<outType> meanArr = createValueArray<outType>(input.dims(), wtdMean);
    Array<outType> diff    = arithOp<outType, af_sub_t>(input, meanArr, input.dims());
    Array<outType> diffSq  = arithOp<outType, af_mul_t>(diff, diff, diff.dims());

    Array<outType> accDiffSq = arithOp<outType, af_mul_t>(diffSq, wts, diffSq.dims());

    outType result = division(reduce_all<af_add_t, outType, outType>(accDiffSq), wtsSum);

    return result;
}

template<typename inType, typename outType>
static af_array var(const af_array& in, const bool isbiased, int dim)
{
    Array<outType> input = cast<outType>(getArray<inType>(in));
    dim4 iDims = input.dims();

    Array<outType> meanArr = mean<outType>(input, dim);

    /* now tile meanArr along dim and use it for variance computation */
    dim4 tileDims(1);
    tileDims[dim] = iDims[dim];
    Array<outType> tMeanArr = tile<outType>(meanArr, tileDims);
    /* now mean array is ready */

    Array<outType> diff    = arithOp<outType, af_sub_t>(input, tMeanArr, tMeanArr.dims());
    Array<outType> diffSq  = arithOp<outType, af_mul_t>(diff, diff, diff.dims());
    Array<outType> redDiff = reduce<af_add_t, outType, outType>(diffSq, dim);
    dim4 oDims = redDiff.dims();

    Array<outType> divArr = createValueArray<outType>(oDims, scalar<outType>(isbiased ? iDims[dim] : iDims[dim]-1));
    Array<outType> result = arithOp<outType, af_div_t>(redDiff, divArr, redDiff.dims());

    return getHandle<outType>(result);
}

template<typename inType, typename outType>
static af_array var(const af_array& in, const af_array& weights, int dim)
{
    typedef typename baseOutType<outType>::type bType;

    Array<outType> input = cast<outType>(getArray<inType>(in));
    Array<outType> wts   = cast<outType>(getArray<bType>(weights));
    dim4 iDims = input.dims();

    Array<outType> meanArr = mean<outType>(input, wts, dim);

    /* now tile meanArr along dim and use it for variance computation */
    dim4 tileDims(1);
    tileDims[dim] = iDims[dim];
    Array<outType> tMeanArr = tile<outType>(meanArr, tileDims);
    /* now mean array is ready */

    Array<outType> diff    = arithOp<outType, af_sub_t>(input, tMeanArr, tMeanArr.dims());
    Array<outType> diffSq  = arithOp<outType, af_mul_t>(diff, diff, diff.dims());
    Array<outType> wDiffSq = arithOp<outType, af_mul_t>(diffSq, wts, diffSq.dims());
    Array<outType> accWDS  = reduce<af_add_t, outType, outType>(wDiffSq, dim);
    Array<outType> divArr  = reduce<af_add_t, outType, outType>(wts, dim);
    Array<outType> result  = arithOp<outType, af_div_t>(accWDS, divArr, accWDS.dims());

    return getHandle<outType>(result);
}

af_err af_var(af_array *out, const af_array in, const bool isbiased, const dim_t dim)
{
    try {
        ARG_ASSERT(2, (dim>=0 && dim<=3));

        af_array output = 0;
        ArrayInfo info = getInfo(in);
        af_dtype type = info.getType();
        switch(type) {
            case f64: output = var<double,  double>(in, isbiased, dim); break;
            case f32: output = var<float ,  float >(in, isbiased, dim); break;
            case s32: output = var<int   ,  float >(in, isbiased, dim); break;
            case u32: output = var<uint  ,  float >(in, isbiased, dim); break;
            case s64: output = var<intl  ,  double>(in, isbiased, dim); break;
            case u64: output = var<uintl ,  double>(in, isbiased, dim); break;
            case  u8: output = var<uchar ,  float >(in, isbiased, dim); break;
            case  b8: output = var<char  ,  float >(in, isbiased, dim); break;
            case c32: output = var<cfloat,  cfloat>(in, isbiased, dim); break;
            case c64: output = var<cdouble,cdouble>(in, isbiased, dim); break;
            default : TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_var_weighted(af_array *out, const af_array in, const af_array weights, const dim_t dim)
{
    try {
        ARG_ASSERT(2, (dim>=0 && dim<=3));

        af_array output = 0;
        ArrayInfo iInfo = getInfo(in);
        ArrayInfo wInfo = getInfo(weights);
        af_dtype iType  = iInfo.getType();
        af_dtype wType  = wInfo.getType();

        ARG_ASSERT(3, (wType==f32 || wType==f64)); /* verify that weights are non-complex real numbers */

        switch(iType) {
            case f64: output = var<double,  double>(in, weights, dim); break;
            case f32: output = var<float ,  float >(in, weights, dim); break;
            case s32: output = var<int   ,  float >(in, weights, dim); break;
            case u32: output = var<uint  ,  float >(in, weights, dim); break;
            case s64: output = var<intl  ,  double>(in, weights, dim); break;
            case u64: output = var<uintl ,  double>(in, weights, dim); break;
            case  u8: output = var<uchar ,  float >(in, weights, dim); break;
            case  b8: output = var<char  ,  float >(in, weights, dim); break;
            case c32: output = var<cfloat,  cfloat>(in, weights, dim); break;
            case c64: output = var<cdouble,cdouble>(in, weights, dim); break;
            default : TYPE_ERROR(1, iType);
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_var_all(double *realVal, double *imagVal, const af_array in, const bool isbiased)
{
    try {
        ArrayInfo info = getInfo(in);
        af_dtype type = info.getType();
        switch(type) {
            case f64: *realVal = varAll<double, double>(in, isbiased); break;
            case f32: *realVal = varAll<float , float >(in, isbiased); break;
            case s32: *realVal = varAll<int   , float >(in, isbiased); break;
            case u32: *realVal = varAll<uint  , float >(in, isbiased); break;
            case s64: *realVal = varAll<intl  , double>(in, isbiased); break;
            case u64: *realVal = varAll<uintl , double>(in, isbiased); break;
            case  u8: *realVal = varAll<uchar , float >(in, isbiased); break;
            case  b8: *realVal = varAll<char  , float >(in, isbiased); break;
            case c32: {
                cfloat tmp = varAll<cfloat,cfloat>(in, isbiased);
                *realVal = real(tmp);
                *imagVal = imag(tmp);
                } break;
            case c64: {
                cdouble tmp = varAll<cdouble,cdouble>(in, isbiased);
                *realVal = real(tmp);
                *imagVal = imag(tmp);
                } break;
            default : TYPE_ERROR(1, type);
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_var_all_weighted(double *realVal, double *imagVal, const af_array in, const af_array weights)
{
    try {
        ArrayInfo iInfo = getInfo(in);
        ArrayInfo wInfo = getInfo(weights);
        af_dtype iType  = iInfo.getType();
        af_dtype wType  = wInfo.getType();

        ARG_ASSERT(3, (wType==f32 || wType==f64)); /* verify that weights are non-complex real numbers */

        switch(iType) {
            case f64: *realVal = varAll<double, double>(in, weights); break;
            case f32: *realVal = varAll<float , float >(in, weights); break;
            case s32: *realVal = varAll<int   , float >(in, weights); break;
            case u32: *realVal = varAll<uint  , float >(in, weights); break;
            case s64: *realVal = varAll<intl  , double >(in, weights); break;
            case u64: *realVal = varAll<uintl , double >(in, weights); break;
            case  u8: *realVal = varAll<uchar , float >(in, weights); break;
            case  b8: *realVal = varAll<char  , float >(in, weights); break;
            case c32: {
                cfloat tmp = varAll<cfloat,cfloat>(in, weights);
                *realVal = real(tmp);
                *imagVal = imag(tmp);
                } break;
            case c64: {
                cdouble tmp = varAll<cdouble,cdouble>(in, weights);
                *realVal = real(tmp);
                *imagVal = imag(tmp);
                } break;
            default : TYPE_ERROR(1, iType);
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}
