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
#include <cmath>

#include "stats.h"

using namespace detail;

template<typename Ti, typename To>
static To corrcoef(const af_array& X, const af_array& Y)
{
    Array<To> xIn = cast<To>(getArray<Ti>(X));
    Array<To> yIn = cast<To>(getArray<Ti>(Y));

    dim4 dims = xIn.dims();
    dim_t n= xIn.elements();

    To xSum = detail::reduce_all<af_add_t, To, To>(xIn);
    To ySum = detail::reduce_all<af_add_t, To, To>(yIn);

    Array<To> xSq = detail::arithOp<To, af_mul_t>(xIn, xIn, dims);
    Array<To> ySq = detail::arithOp<To, af_mul_t>(yIn, yIn, dims);
    Array<To> xy  = detail::arithOp<To, af_mul_t>(xIn, yIn, dims);

    To xSqSum = detail::reduce_all<af_add_t, To, To>(xSq);
    To ySqSum = detail::reduce_all<af_add_t, To, To>(ySq);
    To xySum  = detail::reduce_all<af_add_t, To, To>(xy);

    To result = (n*xySum - xSum*ySum)/(sqrt(n*xSqSum-xSum*xSum)*sqrt(n*ySqSum-ySum*ySum));

    return result;
}

af_err af_corrcoef(double *realVal, double *imagVal, const af_array X, const af_array Y)
{
    try {
        ArrayInfo xInfo = getInfo(X);
        ArrayInfo yInfo = getInfo(Y);
        dim4 xDims      = xInfo.dims();
        dim4 yDims      = yInfo.dims();
        af_dtype xType  = xInfo.getType();
        af_dtype yType  = yInfo.getType();

        ARG_ASSERT(2, (xType==yType));
        ARG_ASSERT(2, (xDims.ndims()==yDims.ndims()));

        for (dim_t i=0; i<xDims.ndims(); ++i)
            ARG_ASSERT(2, (xDims[i]==yDims[i]));

        switch(xType) {
            case f64: *realVal = corrcoef<double, double>(X, Y); break;
            case f32: *realVal = corrcoef<float , float >(X, Y); break;
            case s32: *realVal = corrcoef<int   , float >(X, Y); break;
            case u32: *realVal = corrcoef<uint  , float >(X, Y); break;
            case s64: *realVal = corrcoef<intl  , double>(X, Y); break;
            case u64: *realVal = corrcoef<uintl , double>(X, Y); break;
            case  u8: *realVal = corrcoef<uchar , float >(X, Y); break;
            case  b8: *realVal = corrcoef<char  , float >(X, Y); break;
            default : TYPE_ERROR(1, xType);
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}
