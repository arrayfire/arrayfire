/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <arith.hpp>
#include <backend.hpp>
#include <canny.hpp>
#include <common/err_common.hpp>
#include <complex.hpp>
#include <convolve.hpp>
#include <copy.hpp>
#include <handle.hpp>
#include <histogram.hpp>
#include <iota.hpp>
#include <ireduce.hpp>
#include <logic.hpp>
#include <reduce.hpp>
#include <sobel.hpp>
#include <tile.hpp>
#include <transpose.hpp>
#include <unary.hpp>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/image.h>
#include <af/seq.h>
#include <utility>
#include <vector>

using af::dim4;
using std::vector;

Array<float> gradientMagnitude(const Array<float>& gx, const Array<float>& gy,
                               const bool& isf) {
    if (isf) {
        Array<float> gx2 = detail::abs<float, float>(gx);
        Array<float> gy2 = detail::abs<float, float>(gy);
        return detail::arithOp<float, af_add_t>(gx2, gy2, gx2.dims());
    } else {
        Array<float> gx2 = detail::arithOp<float, af_mul_t>(gx, gx, gx.dims());
        Array<float> gy2 = detail::arithOp<float, af_mul_t>(gy, gy, gy.dims());
        Array<float> sg =
            detail::arithOp<float, af_add_t>(gx2, gy2, gx2.dims());
        return detail::unaryOp<float, af_sqrt_t>(sg);
    }
}

Array<float> otsuThreshold(const Array<float>& supEdges,
                           const unsigned NUM_BINS, const float maxVal) {
    Array<uint> hist =
        detail::histogram<float, uint, false>(supEdges, NUM_BINS, 0, maxVal);

    const af::dim4& hDims = hist.dims();

    // reduce along histogram dimension i.e. 0th dimension
    auto totals = reduce<af_add_t, uint, float>(hist, 0);

    // tile histogram total along 0th dimension
    auto ttotals = tile(totals, af::dim4(hDims[0]));

    // pixel frequency probabilities
    auto probability =
        arithOp<float, af_div_t>(cast<float, uint>(hist), ttotals, hDims);

    std::vector<af_seq> seqBegin(4, af_span);
    std::vector<af_seq> seqRest(4, af_span);

    seqBegin[0] = af_make_seq(0, static_cast<double>(hDims[0] - 1), 1);
    seqRest[0]  = af_make_seq(0, static_cast<double>(hDims[0] - 1), 1);

    const af::dim4& iDims = supEdges.dims();

    Array<float> sigmas = detail::createEmptyArray<float>(hDims);

    for (unsigned b = 0; b < (NUM_BINS - 1); ++b) {
        seqBegin[0].end  = static_cast<double>(b);
        seqRest[0].begin = static_cast<double>(b + 1);

        auto frontPartition = createSubArray(probability, seqBegin, false);
        auto endPartition   = createSubArray(probability, seqRest, false);

        auto qL = reduce<af_add_t, float, float>(frontPartition, 0);
        auto qH = reduce<af_add_t, float, float>(endPartition, 0);

        const dim4 fdims(b + 1, hDims[1], hDims[2], hDims[3]);
        const dim4 edims(NUM_BINS - 1 - b, hDims[1], hDims[2], hDims[3]);

        const dim4 tdims(1, hDims[1], hDims[2], hDims[3]);
        auto frontWeights = iota<float>(dim4(b + 1), tdims);
        auto endWeights   = iota<float>(dim4(NUM_BINS - 1 - b), tdims);
        auto offsetValues = createValueArray<float>(edims, b + 1);

        endWeights = arithOp<float, af_add_t>(endWeights, offsetValues, edims);
        auto __muL =
            arithOp<float, af_mul_t>(frontPartition, frontWeights, fdims);
        auto __muH = arithOp<float, af_mul_t>(endPartition, endWeights, edims);
        auto _muL  = reduce<af_add_t, float, float>(__muL, 0);
        auto _muH  = reduce<af_add_t, float, float>(__muH, 0);
        auto muL   = arithOp<float, af_div_t>(_muL, qL, tdims);
        auto muH   = arithOp<float, af_div_t>(_muH, qH, tdims);
        auto TWOS  = createValueArray<float>(tdims, 2.0f);
        auto diff  = arithOp<float, af_sub_t>(muL, muH, tdims);
        auto sqrd  = arithOp<float, af_pow_t>(diff, TWOS, tdims);
        auto op2   = arithOp<float, af_mul_t>(qL, qH, tdims);
        auto sigma = arithOp<float, af_mul_t>(sqrd, op2, tdims);

        std::vector<af_seq> sliceIndex(4, af_span);
        sliceIndex[0] = {double(b), double(b), 1};

        auto binRes = createSubArray<float>(sigmas, sliceIndex, false);

        copyArray(binRes, sigma);
    }

    dim4 odims          = sigmas.dims();
    odims[0]            = 1;
    Array<float> thresh = createEmptyArray<float>(odims);
    Array<uint> locs    = createEmptyArray<uint>(odims);

    ireduce<af_max_t, float>(thresh, locs, sigmas, 0);

    return cast<float, uint>(tile(locs, dim4(iDims[0], iDims[1], 1, 1)));
}

Array<float> normalize(const Array<float>& supEdges, const float minVal,
                       const float maxVal) {
    auto minArray = createValueArray<float>(supEdges.dims(), minVal);
    auto diff  = arithOp<float, af_sub_t>(supEdges, minArray, supEdges.dims());
    auto denom = createValueArray<float>(supEdges.dims(), (maxVal - minVal));
    return arithOp<float, af_div_t>(diff, denom, supEdges.dims());
}

std::pair<Array<char>, Array<char>> computeCandidates(
    const Array<float>& supEdges, const float t1, const af_canny_threshold ct,
    const float t2) {
    float maxVal  = detail::reduce_all<af_max_t, float, float>(supEdges);
    auto NUM_BINS = static_cast<unsigned>(maxVal);

    auto lowRatio = createValueArray<float>(supEdges.dims(), t1);

    switch (ct) {  // NOLINT(hicpp-multiway-paths-covered)
        case AF_CANNY_THRESHOLD_AUTO_OTSU: {
            auto T2 = otsuThreshold(supEdges, NUM_BINS, maxVal);
            auto T1 = arithOp<float, af_mul_t>(T2, lowRatio, T2.dims());
            Array<char> weak1 =
                logicOp<float, af_ge_t>(supEdges, T1, supEdges.dims());
            Array<char> weak2 =
                logicOp<float, af_lt_t>(supEdges, T2, supEdges.dims());
            Array<char> weak =
                logicOp<char, af_and_t>(weak1, weak2, weak1.dims());
            Array<char> strong =
                logicOp<float, af_ge_t>(supEdges, T2, supEdges.dims());
            return std::make_pair(strong, weak);
        };
        default: {
            float minVal = detail::reduce_all<af_min_t, float, float>(supEdges);
            auto normG   = normalize(supEdges, minVal, maxVal);
            auto T2      = createValueArray<float>(supEdges.dims(), t2);
            auto T1      = createValueArray<float>(supEdges.dims(), t1);
            Array<char> weak1 =
                logicOp<float, af_ge_t>(normG, T1, normG.dims());
            Array<char> weak2 =
                logicOp<float, af_lt_t>(normG, T2, normG.dims());
            Array<char> weak =
                logicOp<char, af_and_t>(weak1, weak2, weak1.dims());
            Array<char> strong =
                logicOp<float, af_ge_t>(normG, T2, normG.dims());
            return std::make_pair(strong, weak);
        };
    }
}

template<typename T>
af_array cannyHelper(const Array<T>& in, const float t1,
                     const af_canny_threshold ct, const float t2,
                     const unsigned sw, const bool isf) {
    static const vector<float> v{-0.11021f, -0.23691f, -0.30576f, -0.23691f,
                                 -0.11021f};
    Array<float> cFilter =
        detail::createHostDataArray<float>(dim4(5, 1), v.data());
    Array<float> rFilter =
        detail::createHostDataArray<float>(dim4(1, 5), v.data());

    // Run separable convolution to smooth the input image
    Array<float> smt = detail::convolve2<float, float, false>(
        cast<float, T>(in), cFilter, rFilter);

    auto g          = detail::sobelDerivatives<float, float>(smt, sw);
    Array<float> gx = g.first;
    Array<float> gy = g.second;

    Array<float> gmag = gradientMagnitude(gx, gy, isf);

    Array<float> supEdges = detail::nonMaximumSuppression(gmag, gx, gy);

    auto swpair = computeCandidates(supEdges, t1, ct, t2);

    return getHandle(
        detail::edgeTrackingByHysteresis(swpair.first, swpair.second));
}

af_err af_canny(af_array* out, const af_array in, const af_canny_threshold ct,
                const float t1, const float t2, const unsigned sw,
                const bool isf) {
    try {
        const ArrayInfo& info = getInfo(in);
        af::dim4 dims         = info.dims();

        DIM_ASSERT(2, (dims.ndims() >= 2));
        // Input should be a minimum of 5x5 image
        // since the gaussian filter used for smoothing
        // the input is of 5x5 size. It's not mandatory but
        // it is essentially of no use if image is less than 5x5
        DIM_ASSERT(2, (dims[0] >= 5 && dims[1] >= 5));
        ARG_ASSERT(5, (sw == 3));

        af_array output;

        af_dtype type = info.getType();
        switch (type) {
            case f32:
                output = cannyHelper<float>(getArray<float>(in), t1, ct, t2, sw,
                                            isf);
                break;
            case f64:
                output = cannyHelper<double>(getArray<double>(in), t1, ct, t2,
                                             sw, isf);
                break;
            case s32:
                output =
                    cannyHelper<int>(getArray<int>(in), t1, ct, t2, sw, isf);
                break;
            case u32:
                output =
                    cannyHelper<uint>(getArray<uint>(in), t1, ct, t2, sw, isf);
                break;
            case s16:
                output = cannyHelper<short>(getArray<short>(in), t1, ct, t2, sw,
                                            isf);
                break;
            case u16:
                output = cannyHelper<ushort>(getArray<ushort>(in), t1, ct, t2,
                                             sw, isf);
                break;
            case u8:
                output = cannyHelper<uchar>(getArray<uchar>(in), t1, ct, t2, sw,
                                            isf);
                break;
            default: TYPE_ERROR(1, type);
        }
        // output array is binary array
        std::swap(output, *out);
    }
    CATCHALL;

    return AF_SUCCESS;
}
