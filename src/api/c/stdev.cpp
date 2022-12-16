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
#include <cmath>
#include <complex>

#include "stats.h"

using af::dim4;
using arrayfire::common::cast;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::createValueArray;
using detail::division;
using detail::intl;
using detail::mean;
using detail::reduce;
using detail::reduce_all;
using detail::scalar;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename inType, typename outType>
static outType stdev(const af_array& in, const af_var_bias bias) {
    using weightType        = typename baseOutType<outType>::type;
    const Array<inType> _in = getArray<inType>(in);
    Array<outType> input    = cast<outType>(_in);
    Array<outType> meanCnst = createValueArray<outType>(
        input.dims(), mean<inType, weightType, outType>(_in));
    Array<outType> diff =
        detail::arithOp<outType, af_sub_t>(input, meanCnst, input.dims());
    Array<outType> diffSq =
        detail::arithOp<outType, af_mul_t>(diff, diff, diff.dims());
    outType result =
        division(reduce_all<af_add_t, outType, outType>(diffSq),
                 (input.elements() - (bias == AF_VARIANCE_SAMPLE)));
    return sqrt(result);
}

template<typename inType, typename outType>
static af_array stdev(const af_array& in, int dim, const af_var_bias bias) {
    using weightType        = typename baseOutType<outType>::type;
    const Array<inType> _in = getArray<inType>(in);
    Array<outType> input    = cast<outType>(_in);
    dim4 iDims              = input.dims();

    Array<outType> meanArr = mean<inType, weightType, outType>(_in, dim);

    /* now tile meanArr along dim and use it for variance computation */
    dim4 tileDims(1);
    tileDims[dim]           = iDims[dim];
    Array<outType> tMeanArr = detail::tile<outType>(meanArr, tileDims);
    /* now mean array is ready */

    Array<outType> diff =
        detail::arithOp<outType, af_sub_t>(input, tMeanArr, tMeanArr.dims());
    Array<outType> diffSq =
        detail::arithOp<outType, af_mul_t>(diff, diff, diff.dims());
    Array<outType> redDiff = reduce<af_add_t, outType, outType>(diffSq, dim);
    const dim4& oDims      = redDiff.dims();

    Array<outType> divArr = createValueArray<outType>(
        oDims, scalar<outType>((iDims[dim] - (bias == AF_VARIANCE_SAMPLE))));
    Array<outType> varArr =
        detail::arithOp<outType, af_div_t>(redDiff, divArr, redDiff.dims());
    Array<outType> result = detail::unaryOp<outType, af_sqrt_t>(varArr);

    return getHandle<outType>(result);
}

// NOLINTNEXTLINE(readability-non-const-parameter)
af_err af_stdev_all(double* realVal, double* imagVal, const af_array in) {
    return af_stdev_all_v2(realVal, imagVal, in, AF_VARIANCE_POPULATION);
}

af_err af_stdev_all_v2(double* realVal, double* imagVal, const af_array in,
                       const af_var_bias bias) {
    UNUSED(imagVal);  // TODO implement for complex values
    try {
        const ArrayInfo& info = getInfo(in);
        af_dtype type         = info.getType();
        switch (type) {
            case f64: *realVal = stdev<double, double>(in, bias); break;
            case f32: *realVal = stdev<float, float>(in, bias); break;
            case s32: *realVal = stdev<int, float>(in, bias); break;
            case u32: *realVal = stdev<uint, float>(in, bias); break;
            case s16: *realVal = stdev<short, float>(in, bias); break;
            case u16: *realVal = stdev<ushort, float>(in, bias); break;
            case s64: *realVal = stdev<intl, double>(in, bias); break;
            case u64: *realVal = stdev<uintl, double>(in, bias); break;
            case u8: *realVal = stdev<uchar, float>(in, bias); break;
            case b8: *realVal = stdev<char, float>(in, bias); break;
            // TODO(umar): FIXME: sqrt(complex) is not present in cuda/opencl
            // backend case c32: {
            //    cfloat tmp = stdev<cfloat,cfloat>(in);
            //    *realVal = real(tmp);
            //    *imagVal = imag(tmp);
            //    } break;
            // case c64: {
            //    cdouble tmp = stdev<cdouble,cdouble>(in);
            //    *realVal = real(tmp);
            //    *imagVal = imag(tmp);
            //    } break;
            default: TYPE_ERROR(1, type);
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_stdev(af_array* out, const af_array in, const dim_t dim) {
    return af_stdev_v2(out, in, AF_VARIANCE_POPULATION, dim);
}

af_err af_stdev_v2(af_array* out, const af_array in, const af_var_bias bias,
                   const dim_t dim) {
    try {
        ARG_ASSERT(2, (dim >= 0 && dim <= 3));

        af_array output       = 0;
        const ArrayInfo& info = getInfo(in);
        af_dtype type         = info.getType();
        switch (type) {
            case f64: output = stdev<double, double>(in, dim, bias); break;
            case f32: output = stdev<float, float>(in, dim, bias); break;
            case s32: output = stdev<int, float>(in, dim, bias); break;
            case u32: output = stdev<uint, float>(in, dim, bias); break;
            case s16: output = stdev<short, float>(in, dim, bias); break;
            case u16: output = stdev<ushort, float>(in, dim, bias); break;
            case s64: output = stdev<intl, double>(in, dim, bias); break;
            case u64: output = stdev<uintl, double>(in, dim, bias); break;
            case u8: output = stdev<uchar, float>(in, dim, bias); break;
            case b8: output = stdev<char, float>(in, dim, bias); break;
            // TODO(umar): FIXME: sqrt(complex) is not present in cuda/opencl
            // backend case c32: output = stdev<cfloat,  cfloat>(in, dim);
            // break; case c64: output = stdev<cdouble,cdouble>(in, dim); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return AF_SUCCESS;
}
