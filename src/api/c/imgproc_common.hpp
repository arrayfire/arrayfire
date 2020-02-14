/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <arith.hpp>
#include <cast.hpp>
#include <logic.hpp>
#include <reduce.hpp>
#include <scan.hpp>

#include <cmath>

namespace common {

template<typename To, typename Ti = To>
detail::Array<To> integralImage(const detail::Array<Ti>& in) {
    auto input               = detail::cast<To, Ti>(in);
    Array<To> horizontalScan = detail::scan<af_add_t, To, To>(input, 0);
    return detail::scan<af_add_t, To, To>(horizontalScan, 1);
}

template<typename T>
detail::Array<T> threshold(const Array<T>& in, T min, T max) {
    const af::dim4 inDims = in.dims();

    auto MN    = createValueArray(inDims, min);
    auto MX    = createValueArray(inDims, max);
    auto below = logicOp<T, af_le_t>(in, MX, inDims);
    auto above = logicOp<T, af_ge_t>(in, MN, inDims);
    auto valid = logicOp<char, af_and_t>(below, above, inDims);

    return arithOp<T, af_mul_t>(in, cast<T, char>(valid), inDims);
}

template<typename To, typename Ti>
detail::Array<To> convRange(const detail::Array<Ti>& in,
                            const To newLow = To(0), const To newHigh = To(1)) {
    auto dims  = in.dims();
    auto input = detail::cast<To, Ti>(in);
    To high    = reduce_all<af_max_t, To, To>(input);
    To low     = reduce_all<af_min_t, To, To>(input);
    To range   = high - low;

    if (std::abs(range) < 1.0e-6) {
        if (low == To(0) && newLow == To(0)) {
            return input;
        } else {
            // Input is constant, use high as constant in converted range
            return createValueArray(dims, newHigh);
        }
    }

    auto minArray = createValueArray(dims, low);
    auto invDen   = createValueArray(dims, To(1.0 / range));
    auto numer    = arithOp<To, af_sub_t>(input, minArray, dims);
    auto result   = arithOp<To, af_mul_t>(numer, invDen, dims);

    if (newLow != To(0) || newHigh != To(1)) {
        To newRange    = newHigh - newLow;
        auto newRngArr = createValueArray(dims, newRange);
        auto newMinArr = createValueArray(dims, newLow);
        auto scaledArr = arithOp<To, af_mul_t>(result, newRngArr, dims);

        result = arithOp<To, af_add_t>(newMinArr, scaledArr, dims);
    }
    return result;
}

}  // namespace common
