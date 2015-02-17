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
#include <tile.hpp>

using namespace detail;

template<typename inType, typename outType>
static outType varAll(const af_array& in, bool isbiased)
{
    Array<outType> *input = cast<outType>(getArray<inType>(in));

    outType mean = division(reduce_all<af_add_t, outType, outType>(*input), input->elements());

    Array<outType> *meanCnst= createValueArray<outType>(input->dims(), mean);

    Array<outType> *diff    = detail::arithOp<outType, af_sub_t>(*input, *meanCnst, input->dims());

    Array<outType> *diffSq  = detail::arithOp<outType, af_mul_t>(*diff, *diff, diff->dims());

    outType result = division(reduce_all<af_add_t, outType, outType>(*diffSq),
        isbiased ? input->elements() : input->elements() - 1);

    destroyArray<outType>(*diffSq);
    destroyArray<outType>(*diff);
    destroyArray<outType>(*meanCnst);
    destroyArray<outType>(*input);

    return result;
}

template<typename T>
using baseOutType = typename std::conditional<  std::is_same<T, cdouble>::value ||
                                                std::is_same<T, double>::value,
                                              double,
                                              float>::type;

template<typename inType, typename outType>
static outType varAll(const af_array& in, const af_array weights)
{
    typedef baseOutType<outType> bType;

    Array<outType> *input = cast<outType>(getArray<inType>(in));
    Array<outType> *wts   = cast<outType>(getArray<bType>(weights));

    dim4 iDims = input->dims();

    Array<outType>* wtdIn= detail::arithOp<outType, af_mul_t>(*input, *wts, iDims);

    outType wtdSum  = reduce_all<af_add_t, outType, outType>(*wtdIn);
    bType wtsSum    = reduce_all<af_add_t, bType, bType>(getArray<bType>(weights));
    outType wtdMean = division(wtdSum, wtsSum);

    Array<outType> *meanArr = createValueArray<outType>(input->dims(), wtdMean);
    Array<outType> *diff    = detail::arithOp<outType, af_sub_t>(*input, *meanArr, input->dims());
    Array<outType> *diffSq  = detail::arithOp<outType, af_mul_t>(*diff, *diff, diff->dims());

    Array<outType> *accDiffSq = detail::arithOp<outType, af_mul_t>(*diffSq, *wts, diffSq->dims());

    outType result = division(reduce_all<af_add_t, outType, outType>(*accDiffSq), wtsSum);

    destroyArray<outType>(*accDiffSq);
    destroyArray<outType>(*diffSq);
    destroyArray<outType>(*diff);
    destroyArray<outType>(*meanArr);
    destroyArray<outType>(*wtdIn);
    destroyArray<outType>(*wts);
    destroyArray<outType>(*input);

    return result;
}

template<typename inType, typename outType>
static af_array var(const af_array& in, bool isbiased, int dim)
{
    Array<outType> *input = cast<outType>(getArray<inType>(in));
    dim4 iDims = input->dims();

    Array<outType> *redArr  = reduce<af_add_t, outType, outType>(*input, dim);
    dim4 oDims = redArr->dims();

    Array<outType> *cnstArr = createValueArray<outType>(oDims, scalar<outType>(iDims[dim]));
    Array<outType> *meanArr = detail::arithOp<outType, af_div_t>(*redArr, *cnstArr, oDims);

    /* now tile meanArr along dim and use it for variance computation */
    dim4 tileDims(1);
    tileDims[dim] = iDims[dim];
    Array<outType> *tMeanArr = detail::tile<outType>(*meanArr, tileDims);
    /* now mean array is ready */

    Array<outType> *diff    = detail::arithOp<outType, af_sub_t>(*input, *tMeanArr, tMeanArr->dims());
    Array<outType> *diffSq  = detail::arithOp<outType, af_mul_t>(*diff, *diff, diff->dims());
    Array<outType> *redDiff = reduce<af_add_t, outType, outType>(*diffSq, dim);
    oDims = redDiff->dims();

    Array<outType> *divArr = createValueArray<outType>(oDims, scalar<outType>(isbiased ? iDims[dim] : iDims[dim]-1));
    Array<outType> *result = detail::arithOp<outType, af_div_t>(*redDiff, *divArr, redDiff->dims());

    destroyArray<outType>(*divArr);
    destroyArray<outType>(*redDiff);
    destroyArray<outType>(*diffSq);
    destroyArray<outType>(*diff);
    destroyArray<outType>(*tMeanArr);
    destroyArray<outType>(*meanArr);
    destroyArray<outType>(*cnstArr);
    destroyArray<outType>(*redArr);
    destroyArray<outType>(*input);

    return getHandle<outType>(*result);
}

template<typename inType, typename outType>
static af_array var(const af_array& in, const af_array& weights, dim_type dim)
{
    typedef baseOutType<outType> bType;

    Array<outType> *input = cast<outType>(getArray<inType>(in));
    Array<outType> *wts   = cast<outType>(getArray<bType>(weights));
    dim4 iDims = input->dims();

    Array<outType> *prodArr = detail::arithOp<outType, af_mul_t>(*input, *wts, iDims);
    Array<outType> *sumArr  = reduce<af_add_t, outType, outType>(*prodArr, dim);
    Array<outType> *wSumArr = reduce<af_add_t, outType, outType>(*wts, dim);
    dim4 oDims = sumArr->dims();

    Array<outType> *meanArr   = detail::arithOp<outType, af_div_t>(*sumArr, *wSumArr, oDims);

    /* now tile meanArr along dim and use it for variance computation */
    dim4 tileDims(1);
    tileDims[dim] = iDims[dim];
    Array<outType> *tMeanArr = detail::tile<outType>(*meanArr, tileDims);
    /* now mean array is ready */

    Array<outType> *diff    = detail::arithOp<outType, af_sub_t>(*input, *tMeanArr, tMeanArr->dims());
    Array<outType> *diffSq  = detail::arithOp<outType, af_mul_t>(*diff, *diff, diff->dims());
    Array<outType> *wDiffSq = detail::arithOp<outType, af_mul_t>(*diffSq, *wts, diffSq->dims());
    Array<outType> *accWDS  = reduce<af_add_t, outType, outType>(*wDiffSq, dim);
    Array<outType> *divArr  = reduce<af_add_t, outType, outType>(*wts, dim);
    Array<outType> *result  = detail::arithOp<outType, af_div_t>(*accWDS, *divArr, accWDS->dims());

    destroyArray<outType>(*input);
    destroyArray<outType>(*wts);
    destroyArray<outType>(*prodArr);
    destroyArray<outType>(*sumArr);
    destroyArray<outType>(*wSumArr);
    destroyArray<outType>(*meanArr);
    destroyArray<outType>(*tMeanArr);
    destroyArray<outType>(*diff);
    destroyArray<outType>(*diffSq);
    destroyArray<outType>(*wDiffSq);
    destroyArray<outType>(*accWDS);
    destroyArray<outType>(*divArr);

    return getHandle<outType>(*result);
}

af_err af_var(af_array *out, const af_array in, bool isbiased, dim_type dim)
{
    try {
        af_array output = 0;
        ArrayInfo info = getInfo(in);
        af_dtype type = info.getType();
        switch(type) {
            case f64: output = var<double,  double>(in, isbiased, dim); break;
            case f32: output = var<float ,  float >(in, isbiased, dim); break;
            case s32: output = var<int   ,  float >(in, isbiased, dim); break;
            case u32: output = var<uint  ,  float >(in, isbiased, dim); break;
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

af_err af_var_weighted(af_array *out, const af_array in, const af_array weights, dim_type dim)
{
    try {
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

af_err af_var_all(double *realVal, double *imagVal, const af_array in, bool isbiased)
{
    try {
        ArrayInfo info = getInfo(in);
        af_dtype type = info.getType();
        switch(type) {
            case f64: *realVal = varAll<double, double>(in, isbiased); break;
            case f32: *realVal = varAll<float , float >(in, isbiased); break;
            case s32: *realVal = varAll<int   , float >(in, isbiased); break;
            case u32: *realVal = varAll<uint  , float >(in, isbiased); break;
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
