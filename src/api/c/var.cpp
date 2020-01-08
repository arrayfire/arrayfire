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
#include <cast.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <math.hpp>
#include <mean.hpp>
#include <reduce.hpp>
#include <tile.hpp>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/statistics.h>

#include "stats.h"

#include <tuple>

using namespace detail;

using std::ignore;
using std::make_tuple;
using std::tie;
using std::tuple;

template<typename inType, typename outType>
static outType varAll(const af_array& in, const bool isbiased) {
    typedef typename baseOutType<outType>::type weightType;
    Array<inType> inArr  = getArray<inType>(in);
    Array<outType> input = cast<outType>(inArr);

    Array<outType> meanCnst = createValueArray<outType>(
        input.dims(), mean<inType, weightType, outType>(inArr));

    Array<outType> diff =
        arithOp<outType, af_sub_t>(input, meanCnst, input.dims());

    Array<outType> diffSq = arithOp<outType, af_mul_t>(diff, diff, diff.dims());

    outType result =
        division(reduce_all<af_add_t, outType, outType>(diffSq),
                 isbiased ? input.elements() : input.elements() - 1);

    return result;
}

template<typename inType, typename outType>
static outType varAll(const af_array& in, const af_array weights) {
    typedef typename baseOutType<outType>::type bType;

    Array<outType> input = cast<outType>(getArray<inType>(in));
    Array<outType> wts   = cast<outType>(getArray<bType>(weights));

    bType wtsSum = reduce_all<af_add_t, bType, bType>(getArray<bType>(weights));
    outType wtdMean = mean<outType, bType>(input, getArray<bType>(weights));

    Array<outType> meanArr = createValueArray<outType>(input.dims(), wtdMean);
    Array<outType> diff =
        arithOp<outType, af_sub_t>(input, meanArr, input.dims());
    Array<outType> diffSq = arithOp<outType, af_mul_t>(diff, diff, diff.dims());

    Array<outType> accDiffSq =
        arithOp<outType, af_mul_t>(diffSq, wts, diffSq.dims());

    outType result =
        division(reduce_all<af_add_t, outType, outType>(accDiffSq), wtsSum);

    return result;
}

template<typename inType, typename outType>
static tuple<Array<outType>, Array<outType>> meanvar(
    const Array<inType>& in,
    const Array<typename baseOutType<outType>::type>& weights,
    const af_var_bias bias, const dim_t dim) {
    typedef typename baseOutType<outType>::type weightType;
    Array<outType> input = cast<outType>(in);
    dim4 iDims           = input.dims();

    Array<outType> meanArr = createEmptyArray<outType>({0});
    Array<outType> normArr = createEmptyArray<outType>({0});
    if (weights.isEmpty()) {
        meanArr  = mean<outType, weightType, outType>(input, dim);
        auto val = 1.0 / (bias == AF_VARIANCE_POPULATION ? iDims[dim]
                                                         : iDims[dim] - 1);
        normArr =
            createValueArray<outType>(meanArr.dims(), scalar<outType>(val));
    } else {
        meanArr               = mean<outType, weightType>(input, weights, dim);
        Array<outType> wtsSum = cast<outType>(
            reduce<af_add_t, weightType, weightType>(weights, dim));
        Array<outType> ones =
            createValueArray<outType>(wtsSum.dims(), scalar<outType>(1));
        if (bias == AF_VARIANCE_SAMPLE) {
            wtsSum = arithOp<outType, af_sub_t>(wtsSum, ones, ones.dims());
        }
        normArr = arithOp<outType, af_div_t>(ones, wtsSum, meanArr.dims());
    }

    /* now tile meanArr along dim and use it for variance computation */
    dim4 tileDims(1);
    tileDims[dim]           = iDims[dim];
    Array<outType> tMeanArr = tile<outType>(meanArr, tileDims);
    /* now mean array is ready */

    Array<outType> diff =
        arithOp<outType, af_sub_t>(input, tMeanArr, tMeanArr.dims());
    Array<outType> diffSq = arithOp<outType, af_mul_t>(diff, diff, diff.dims());
    Array<outType> redDiff = reduce<af_add_t, outType, outType>(diffSq, dim);

    Array<outType> variance =
        arithOp<outType, af_mul_t>(normArr, redDiff, redDiff.dims());

    return make_tuple(meanArr, variance);
}

template<typename inType, typename outType>
static tuple<af_array, af_array> meanvar(const af_array& in,
                                         const af_array& weights,
                                         const af_var_bias bias,
                                         const dim_t dim) {
    typedef typename baseOutType<outType>::type weightType;
    Array<outType> mean = createEmptyArray<outType>({0}),
                   var  = createEmptyArray<outType>({0});

    Array<weightType> w = createEmptyArray<weightType>({0});
    if (weights != 0) { w = getArray<weightType>(weights); }
    tie(mean, var) =
        meanvar<inType, outType>(getArray<inType>(in), w, bias, dim);
    return make_tuple(getHandle(mean), getHandle(var));
}

/// Calculates the variance
///
/// \note Only calculates the weighted variance if the weights array is
/// non-empty
template<typename inType, typename outType>
static Array<outType> var(
    const Array<inType>& in,
    const Array<typename baseOutType<outType>::type>& weights,
    const af_var_bias bias, int dim) {
    Array<outType> variance = createEmptyArray<outType>({0});
    tie(ignore, variance)   = meanvar<inType, outType>(in, weights, bias, dim);
    return variance;
}

template<typename inType, typename outType>
static af_array var_(const af_array& in, const af_array& weights,
                     const af_var_bias bias, int dim) {
    using bType = typename baseOutType<outType>::type;
    if (weights == 0) {
        Array<bType> empty = createEmptyArray<bType>({0});
        return getHandle(
            var<inType, outType>(getArray<inType>(in), empty, bias, dim));
    } else {
        return getHandle(var<inType, outType>(
            getArray<inType>(in), getArray<bType>(weights), bias, dim));
    }
}

af_err af_var(af_array* out, const af_array in, const bool isbiased,
              const dim_t dim) {
    try {
        ARG_ASSERT(3, (dim >= 0 && dim <= 3));

        af_array output       = 0;
        const ArrayInfo& info = getInfo(in);
        af_dtype type         = info.getType();

        af_array no_weights = 0;
        af_var_bias bias =
            (isbiased) ? AF_VARIANCE_SAMPLE: AF_VARIANCE_POPULATION;
        switch (type) {
            case f32:
                output = var_<float, float>(in, no_weights, bias, dim);
                break;
            case f64:
                output = var_<double, double>(in, no_weights, bias, dim);
                break;
            case s32:
                output = var_<int, float>(in, no_weights, bias, dim);
                break;
            case u32:
                output = var_<uint, float>(in, no_weights, bias, dim);
                break;
            case s16:
                output = var_<short, float>(in, no_weights, bias, dim);
                break;
            case u16:
                output = var_<ushort, float>(in, no_weights, bias, dim);
                break;
            case s64:
                output = var_<intl, double>(in, no_weights, bias, dim);
                break;
            case u64:
                output = var_<uintl, double>(in, no_weights, bias, dim);
                break;
            case u8:
                output = var_<uchar, float>(in, no_weights, bias, dim);
                break;
            case b8:
                output = var_<char, float>(in, no_weights, bias, dim);
                break;
            case c32:
                output = var_<cfloat, cfloat>(in, no_weights, bias, dim);
                break;
            case c64:
                output = var_<cdouble, cdouble>(in, no_weights, bias, dim);
                break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_var_weighted(af_array* out, const af_array in, const af_array weights,
                       const dim_t dim) {
    try {
        ARG_ASSERT(3, (dim >= 0 && dim <= 3));

        af_array output        = 0;
        const ArrayInfo& iInfo = getInfo(in);
        const ArrayInfo& wInfo = getInfo(weights);
        af_dtype iType         = iInfo.getType();
        af_dtype wType         = wInfo.getType();

        ARG_ASSERT(
            2,
            (wType == f32 ||
             wType ==
                 f64)); /* verify that weights are non-complex real numbers */

        switch (iType) {
            case f64:
                output = var_<double, double>(in, weights,
                                              AF_VARIANCE_POPULATION, dim);
                break;
            case f32:
                output = var_<float, float>(in, weights, AF_VARIANCE_POPULATION,
                                            dim);
                break;
            case s32:
                output =
                    var_<int, float>(in, weights, AF_VARIANCE_POPULATION, dim);
                break;
            case u32:
                output =
                    var_<uint, float>(in, weights, AF_VARIANCE_POPULATION, dim);
                break;
            case s16:
                output = var_<short, float>(in, weights, AF_VARIANCE_POPULATION,
                                            dim);
                break;
            case u16:
                output = var_<ushort, float>(in, weights,
                                             AF_VARIANCE_POPULATION, dim);
                break;
            case s64:
                output = var_<intl, double>(in, weights, AF_VARIANCE_POPULATION,
                                            dim);
                break;
            case u64:
                output = var_<uintl, double>(in, weights,
                                             AF_VARIANCE_POPULATION, dim);
                break;
            case u8:
                output = var_<uchar, float>(in, weights, AF_VARIANCE_POPULATION,
                                            dim);
                break;
            case b8:
                output =
                    var_<char, float>(in, weights, AF_VARIANCE_POPULATION, dim);
                break;
            case c32:
                output = var_<cfloat, cfloat>(in, weights,
                                              AF_VARIANCE_POPULATION, dim);
                break;
            case c64:
                output = var_<cdouble, cdouble>(in, weights,
                                                AF_VARIANCE_POPULATION, dim);
                break;
            default: TYPE_ERROR(1, iType);
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_var_all(double* realVal, double* imagVal, const af_array in,
                  const bool isbiased) {
    try {
        const ArrayInfo& info = getInfo(in);
        af_dtype type         = info.getType();
        switch (type) {
            case f64: *realVal = varAll<double, double>(in, isbiased); break;
            case f32: *realVal = varAll<float, float>(in, isbiased); break;
            case s32: *realVal = varAll<int, float>(in, isbiased); break;
            case u32: *realVal = varAll<uint, float>(in, isbiased); break;
            case s16: *realVal = varAll<short, float>(in, isbiased); break;
            case u16: *realVal = varAll<ushort, float>(in, isbiased); break;
            case s64: *realVal = varAll<intl, double>(in, isbiased); break;
            case u64: *realVal = varAll<uintl, double>(in, isbiased); break;
            case u8: *realVal = varAll<uchar, float>(in, isbiased); break;
            case b8: *realVal = varAll<char, float>(in, isbiased); break;
            case c32: {
                cfloat tmp = varAll<cfloat, cfloat>(in, isbiased);
                *realVal   = real(tmp);
                *imagVal   = imag(tmp);
            } break;
            case c64: {
                cdouble tmp = varAll<cdouble, cdouble>(in, isbiased);
                *realVal    = real(tmp);
                *imagVal    = imag(tmp);
            } break;
            default: TYPE_ERROR(1, type);
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_var_all_weighted(double* realVal, double* imagVal, const af_array in,
                           const af_array weights) {
    try {
        const ArrayInfo& iInfo = getInfo(in);
        const ArrayInfo& wInfo = getInfo(weights);
        af_dtype iType         = iInfo.getType();
        af_dtype wType         = wInfo.getType();

        ARG_ASSERT(
            3,
            (wType == f32 ||
             wType ==
                 f64)); /* verify that weights are non-complex real numbers */

        switch (iType) {
            case f64: *realVal = varAll<double, double>(in, weights); break;
            case f32: *realVal = varAll<float, float>(in, weights); break;
            case s32: *realVal = varAll<int, float>(in, weights); break;
            case u32: *realVal = varAll<uint, float>(in, weights); break;
            case s16: *realVal = varAll<short, float>(in, weights); break;
            case u16: *realVal = varAll<ushort, float>(in, weights); break;
            case s64: *realVal = varAll<intl, double>(in, weights); break;
            case u64: *realVal = varAll<uintl, double>(in, weights); break;
            case u8: *realVal = varAll<uchar, float>(in, weights); break;
            case b8: *realVal = varAll<char, float>(in, weights); break;
            case c32: {
                cfloat tmp = varAll<cfloat, cfloat>(in, weights);
                *realVal   = real(tmp);
                *imagVal   = imag(tmp);
            } break;
            case c64: {
                cdouble tmp = varAll<cdouble, cdouble>(in, weights);
                *realVal    = real(tmp);
                *imagVal    = imag(tmp);
            } break;
            default: TYPE_ERROR(1, iType);
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_meanvar(af_array* mean, af_array* var, const af_array in,
                  const af_array weights, const af_var_bias bias,
                  const dim_t dim) {
    try {
        const ArrayInfo& iInfo = getInfo(in);
        if (weights != 0) {
            const ArrayInfo& wInfo = getInfo(weights);
            af_dtype wType         = wInfo.getType();
            ARG_ASSERT(3, (wType == f32 || wType == f64));
        }
        af_dtype iType = iInfo.getType();

        switch (iType) {
            case f32:
                tie(*mean, *var) =
                    meanvar<float, float>(in, weights, bias, dim);
                break;
            case f64:
                tie(*mean, *var) =
                    meanvar<double, double>(in, weights, bias, dim);
                break;
            case s32:
                tie(*mean, *var) = meanvar<int, float>(in, weights, bias, dim);
                break;
            case u32:
                tie(*mean, *var) = meanvar<uint, float>(in, weights, bias, dim);
                break;
            case s16:
                tie(*mean, *var) =
                    meanvar<short, float>(in, weights, bias, dim);
                break;
            case u16:
                tie(*mean, *var) =
                    meanvar<ushort, float>(in, weights, bias, dim);
                break;
            case s64:
                tie(*mean, *var) =
                    meanvar<intl, double>(in, weights, bias, dim);
                break;
            case u64:
                tie(*mean, *var) =
                    meanvar<uintl, double>(in, weights, bias, dim);
                break;
            case u8:
                tie(*mean, *var) =
                    meanvar<uchar, float>(in, weights, bias, dim);
                break;
            case b8:
                tie(*mean, *var) = meanvar<char, float>(in, weights, bias, dim);
                break;
            case c32:
                tie(*mean, *var) =
                    meanvar<cfloat, cfloat>(in, weights, bias, dim);
                break;
            case c64:
                tie(*mean, *var) =
                    meanvar<cdouble, cdouble>(in, weights, bias, dim);
                break;
            default: TYPE_ERROR(1, iType);
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}
