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
#include <unary.hpp>
#include <math.hpp>
#include <cast.hpp>
#include <tile.hpp>
#include <cmath>
#include <complex>

#include "stats.h"

using namespace detail;

template<typename inType, typename outType>
static outType stdev(const af_array& in)
{
    Array<outType> input = cast<outType>(getArray<inType>(in));

    Array<outType> meanCnst= createValueArray<outType>(input.dims(), mean<outType>(input));

    Array<outType> diff    = detail::arithOp<outType, af_sub_t>(input, meanCnst, input.dims());

    Array<outType> diffSq  = detail::arithOp<outType, af_mul_t>(diff, diff, diff.dims());

    outType result = division(reduce_all<af_add_t, outType, outType>(diffSq), input.elements());

    return sqrt(result);
}

template<typename inType, typename outType>
static af_array stdev(const af_array& in, int dim)
{
    Array<outType> input = cast<outType>(getArray<inType>(in));
    dim4 iDims = input.dims();

    Array<outType> meanArr = mean<outType>(input, dim);

    /* now tile meanArr along dim and use it for variance computation */
    dim4 tileDims(1);
    tileDims[dim] = iDims[dim];
    Array<outType> tMeanArr = detail::tile<outType>(meanArr, tileDims);
    /* now mean array is ready */

    Array<outType> diff    = detail::arithOp<outType, af_sub_t>(input, tMeanArr, tMeanArr.dims());
    Array<outType> diffSq  = detail::arithOp<outType, af_mul_t>(diff, diff, diff.dims());
    Array<outType> redDiff = reduce<af_add_t, outType, outType>(diffSq, dim);
    dim4 oDims = redDiff.dims();

    Array<outType> divArr = createValueArray<outType>(oDims, scalar<outType>(iDims[dim]));
    Array<outType> varArr = detail::arithOp<outType, af_div_t>(redDiff, divArr, redDiff.dims());
    Array<outType> result = detail::unaryOp<outType, af_sqrt_t>(varArr);

    return getHandle<outType>(result);
}

af_err af_stdev_all(double *realVal, double *imagVal, const af_array in)
{
    try {
        ArrayInfo info = getInfo(in);
        af_dtype type = info.getType();
        switch(type) {
            case f64: *realVal = stdev<double, double>(in); break;
            case f32: *realVal = stdev<float , float >(in); break;
            case s32: *realVal = stdev<int   , float >(in); break;
            case u32: *realVal = stdev<uint  , float >(in); break;
            case s64: *realVal = stdev<intl  , double>(in); break;
            case u64: *realVal = stdev<uintl , double>(in); break;
            case  u8: *realVal = stdev<uchar , float >(in); break;
            case  b8: *realVal = stdev<char  , float >(in); break;
            // TODO: FIXME: sqrt(complex) is not present in cuda/opencl backend
            //case c32: {
            //    cfloat tmp = stdev<cfloat,cfloat>(in);
            //    *realVal = real(tmp);
            //    *imagVal = imag(tmp);
            //    } break;
            //case c64: {
            //    cdouble tmp = stdev<cdouble,cdouble>(in);
            //    *realVal = real(tmp);
            //    *imagVal = imag(tmp);
            //    } break;
            default : TYPE_ERROR(1, type);
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_stdev(af_array *out, const af_array in, const dim_t dim)
{
    try {
        ARG_ASSERT(2, (dim>=0 && dim<=3));

        af_array output = 0;
        ArrayInfo info = getInfo(in);
        af_dtype type = info.getType();
        switch(type) {
            case f64: output = stdev<double,  double>(in, dim); break;
            case f32: output = stdev<float ,  float >(in, dim); break;
            case s32: output = stdev<int   ,  float >(in, dim); break;
            case u32: output = stdev<uint  ,  float >(in, dim); break;
            case s64: output = stdev<intl  ,  double>(in, dim); break;
            case u64: output = stdev<uintl ,  double>(in, dim); break;
            case  u8: output = stdev<uchar ,  float >(in, dim); break;
            case  b8: output = stdev<char  ,  float >(in, dim); break;
            // TODO: FIXME: sqrt(complex) is not present in cuda/opencl backend
            //case c32: output = stdev<cfloat,  cfloat>(in, dim); break;
            //case c64: output = stdev<cdouble,cdouble>(in, dim); break;
            default : TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return AF_SUCCESS;
}
