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
#include <common/err_common.hpp>
#include <handle.hpp>
#include <math.hpp>
#include <reduce.hpp>
#include <stats.h>
#include <types.hpp>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/statistics.h>

#include <cmath>

using af::dim4;
using arrayfire::common::cast;
using detail::arithOp;
using detail::Array;
using detail::intl;
using detail::reduce_all;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename Ti, typename To>
static To corrcoef(const af_array& X, const af_array& Y) {
    Array<To> xIn = cast<To>(getArray<Ti>(X));
    Array<To> yIn = cast<To>(getArray<Ti>(Y));

    const dim4& dims = xIn.dims();
    dim_t n          = xIn.elements();

    To xSum = reduce_all<af_add_t, To, To>(xIn);
    To ySum = reduce_all<af_add_t, To, To>(yIn);

    Array<To> xSq = arithOp<To, af_mul_t>(xIn, xIn, dims);
    Array<To> ySq = arithOp<To, af_mul_t>(yIn, yIn, dims);
    Array<To> xy  = arithOp<To, af_mul_t>(xIn, yIn, dims);

    To xSqSum = reduce_all<af_add_t, To, To>(xSq);
    To ySqSum = reduce_all<af_add_t, To, To>(ySq);
    To xySum  = reduce_all<af_add_t, To, To>(xy);

    To result =
        (n * xySum - xSum * ySum) / (std::sqrt(n * xSqSum - xSum * xSum) *
                                     std::sqrt(n * ySqSum - ySum * ySum));

    return result;
}

// NOLINTNEXTLINE
af_err af_corrcoef(double* realVal, double* imagVal, const af_array X,
                   const af_array Y) {
    UNUSED(imagVal);  // TODO(umar): implement for complex types
    try {
        const ArrayInfo& xInfo = getInfo(X);
        const ArrayInfo& yInfo = getInfo(Y);
        dim4 xDims             = xInfo.dims();
        dim4 yDims             = yInfo.dims();
        af_dtype xType         = xInfo.getType();
        af_dtype yType         = yInfo.getType();

        ARG_ASSERT(2, (xType == yType));
        ARG_ASSERT(2, (xDims.ndims() == yDims.ndims()));

        for (dim_t i = 0; i < xDims.ndims(); ++i) {
            ARG_ASSERT(2, (xDims[i] == yDims[i]));
        }

        switch (xType) {
            case f64: *realVal = corrcoef<double, double>(X, Y); break;
            case f32: *realVal = corrcoef<float, float>(X, Y); break;
            case s32: *realVal = corrcoef<int, float>(X, Y); break;
            case u32: *realVal = corrcoef<uint, float>(X, Y); break;
            case s64: *realVal = corrcoef<intl, double>(X, Y); break;
            case u64: *realVal = corrcoef<uintl, double>(X, Y); break;
            case s16: *realVal = corrcoef<short, float>(X, Y); break;
            case u16: *realVal = corrcoef<ushort, float>(X, Y); break;
            case u8: *realVal = corrcoef<uchar, float>(X, Y); break;
            case b8: *realVal = corrcoef<char, float>(X, Y); break;
            default: TYPE_ERROR(1, xType);
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}
