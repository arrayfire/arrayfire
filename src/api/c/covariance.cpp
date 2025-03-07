/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arith.hpp>
#include <backend.hpp>
#include <common/cast.hpp>
#include <handle.hpp>
#include <math.hpp>
#include <mean.hpp>
#include <reduce.hpp>
#include <tile.hpp>
#include <unary.hpp>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/statistics.h>

#include "stats.h"

using af::dim4;
using arrayfire::common::cast;
using detail::arithOp;
using detail::Array;
using detail::createValueArray;
using detail::intl;
using detail::mean;
using detail::reduce;
using detail::scalar;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename T, typename cType>
static af_array cov(const af_array& X, const af_array& Y,
                    const af_var_bias bias) {
    using weightType  = typename baseOutType<cType>::type;
    const Array<T> _x = getArray<T>(X);
    const Array<T> _y = getArray<T>(Y);
    Array<cType> xArr = cast<cType>(_x);
    Array<cType> yArr = cast<cType>(_y);

    dim4 xDims = xArr.dims();
    dim_t N    = (bias == AF_VARIANCE_SAMPLE ? xDims[0] - 1 : xDims[0]);

    Array<cType> xmArr =
        createValueArray<cType>(xDims, mean<T, weightType, cType>(_x));
    Array<cType> ymArr =
        createValueArray<cType>(xDims, mean<T, weightType, cType>(_y));
    Array<cType> nArr = createValueArray<cType>(xDims, scalar<cType>(N));

    Array<cType> diffX  = arithOp<cType, af_sub_t>(xArr, xmArr, xDims);
    Array<cType> diffY  = arithOp<cType, af_sub_t>(yArr, ymArr, xDims);
    Array<cType> mulXY  = arithOp<cType, af_mul_t>(diffX, diffY, xDims);
    Array<cType> redArr = reduce<af_add_t, cType, cType>(mulXY, 0);
    xDims[0]            = 1;
    Array<cType> result = arithOp<cType, af_div_t>(redArr, nArr, xDims);

    return getHandle<cType>(result);
}

af_err af_cov(af_array* out, const af_array X, const af_array Y,
              const bool isbiased) {
    const af_var_bias bias =
        (isbiased ? AF_VARIANCE_SAMPLE : AF_VARIANCE_POPULATION);
    return af_cov_v2(out, X, Y, bias);
}

af_err af_cov_v2(af_array* out, const af_array X, const af_array Y,
                 const af_var_bias bias) {
    try {
        const ArrayInfo& xInfo = getInfo(X);
        const ArrayInfo& yInfo = getInfo(Y);
        dim4 xDims             = xInfo.dims();
        dim4 yDims             = yInfo.dims();
        af_dtype xType         = xInfo.getType();
        af_dtype yType         = yInfo.getType();

        ARG_ASSERT(1, (xDims.ndims() <= 2));
        ARG_ASSERT(2, (xDims.ndims() == yDims.ndims()));
        ARG_ASSERT(2, (xDims[0] == yDims[0]));
        ARG_ASSERT(2, (xDims[1] == yDims[1]));
        ARG_ASSERT(2, (xType == yType));

        af_array output = 0;
        switch (xType) {
            case f64: output = cov<double, double>(X, Y, bias); break;
            case f32: output = cov<float, float>(X, Y, bias); break;
            case s32: output = cov<int, float>(X, Y, bias); break;
            case u32: output = cov<uint, float>(X, Y, bias); break;
            case s64: output = cov<intl, double>(X, Y, bias); break;
            case u64: output = cov<uintl, double>(X, Y, bias); break;
            case s16: output = cov<short, float>(X, Y, bias); break;
            case u16: output = cov<ushort, float>(X, Y, bias); break;
            case u8: output = cov<uchar, float>(X, Y, bias); break;
            default: TYPE_ERROR(1, xType);
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return AF_SUCCESS;
}
