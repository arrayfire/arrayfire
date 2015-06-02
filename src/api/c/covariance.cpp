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

#include "stats.h"

using af::dim4;
using namespace detail;

template<typename T, typename cType>
static af_array cov(const af_array& X, const af_array& Y, const bool isbiased)
{
    Array<cType> xArr = cast<cType>(getArray<T>(X));
    Array<cType> yArr = cast<cType>(getArray<T>(Y));

    dim4 xDims = xArr.dims();
    dim_t N = isbiased ? xDims[0] : xDims[0]-1;

    Array<cType> xmArr = createValueArray<cType>(xDims, mean<cType>(xArr));
    Array<cType> ymArr = createValueArray<cType>(xDims, mean<cType>(yArr));
    Array<cType> nArr  = createValueArray<cType>(xDims, scalar<cType>(N));

    Array<cType> diffX = detail::arithOp<cType, af_sub_t>(xArr, xmArr, xDims);
    Array<cType> diffY = detail::arithOp<cType, af_sub_t>(yArr, ymArr, xDims);
    Array<cType> mulXY = detail::arithOp<cType, af_mul_t>(diffX, diffY, xDims);
    Array<cType> redArr= detail::reduce<af_add_t, cType, cType>(mulXY, 0);
    xDims[0] = 1;
    Array<cType> result= detail::arithOp<cType, af_div_t>(redArr, nArr, xDims);

    return getHandle<cType>(result);
}

af_err af_cov(af_array* out, const af_array X, const af_array Y, const bool isbiased)
{
    try {
        ArrayInfo xInfo = getInfo(X);
        ArrayInfo yInfo = getInfo(Y);
        dim4 xDims      = xInfo.dims();
        dim4 yDims      = yInfo.dims();
        af_dtype xType  = xInfo.getType();
        af_dtype yType  = yInfo.getType();

        ARG_ASSERT(1, (xDims.ndims()<=2));
        ARG_ASSERT(2, (xDims.ndims()==yDims.ndims()));
        ARG_ASSERT(2, (xDims[0]==yDims[0]));
        ARG_ASSERT(2, (xDims[1]==yDims[1]));
        ARG_ASSERT(2, (xType==yType));

        af_array output = 0;
        switch(xType) {
            case f64: output = cov<double, double>(X, Y, isbiased); break;
            case f32: output = cov<float , float >(X, Y, isbiased); break;
            case s32: output = cov<int   , float >(X, Y, isbiased); break;
            case u32: output = cov<uint  , float >(X, Y, isbiased); break;
            case s64: output = cov<intl  , double>(X, Y, isbiased); break;
            case u64: output = cov<uintl , double>(X, Y, isbiased); break;
            case  u8: output = cov<uchar , float >(X, Y, isbiased); break;
            default : TYPE_ERROR(1, xType);
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return AF_SUCCESS;
}
