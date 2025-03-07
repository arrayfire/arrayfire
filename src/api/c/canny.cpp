/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <canny.hpp>

#include <Array.hpp>
#include <arith.hpp>
#include <backend.hpp>
#include <common/cast.hpp>
#include <common/err_common.hpp>
#include <common/tile.hpp>
#include <complex.hpp>
#include <convolve.hpp>
#include <copy.hpp>
#include <handle.hpp>
#include <histogram.hpp>
#include <iota.hpp>
#include <ireduce.hpp>
#include <logic.hpp>
#include <reduce.hpp>
#include <scan.hpp>
#include <sobel.hpp>
#include <transpose.hpp>
#include <unary.hpp>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/image.h>
#include <af/seq.h>
#include <utility>
#include <vector>

using af::dim4;
using arrayfire::common::cast;
using arrayfire::common::tile;
using detail::arithOp;
using detail::Array;
using detail::convolve2;
using detail::createEmptyArray;
using detail::createHostDataArray;
using detail::createSubArray;
using detail::createValueArray;
using detail::getScalar;
using detail::histogram;
using detail::iota;
using detail::ireduce;
using detail::logicOp;
using detail::reduce;
using detail::reduce_all;
using detail::scan;
using detail::sobelDerivatives;
using detail::uchar;
using detail::uint;
using detail::unaryOp;
using detail::ushort;
using std::make_pair;
using std::pair;
using std::vector;

namespace {
Array<float> gradientMagnitude(const Array<float>& gx, const Array<float>& gy,
                               const bool& isf) {
    using detail::abs;
    if (isf) {
        Array<float> gx2 = abs<float, float>(gx);
        Array<float> gy2 = abs<float, float>(gy);
        return arithOp<float, af_add_t>(gx2, gy2, gx2.dims());
    } else {
        Array<float> gx2 = arithOp<float, af_mul_t>(gx, gx, gx.dims());
        Array<float> gy2 = arithOp<float, af_mul_t>(gy, gy, gy.dims());
        Array<float> sg  = arithOp<float, af_add_t>(gx2, gy2, gx2.dims());
        return unaryOp<float, af_sqrt_t>(sg);
    }
}

Array<float> otsuThreshold(const Array<float>& in, const unsigned NUM_BINS,
                           const float maxVal) {
    Array<uint> hist = histogram<float>(in, NUM_BINS, 0, maxVal, false);

    const dim4& inDims = in.dims();
    const dim4& hDims  = hist.dims();

    const dim4 oDims(1, hDims[1], hDims[2], hDims[3]);
    vector<af_seq> seqBegin(4, af_span);
    vector<af_seq> seqRest(4, af_span);
    vector<af_seq> sliceIndex(4, af_span);

    seqBegin[0] = af_make_seq(0, static_cast<double>(hDims[0] - 1), 1);
    seqRest[0]  = af_make_seq(0, static_cast<double>(hDims[0] - 1), 1);

    Array<float> UnitP  = createValueArray<float>(oDims, 1.0f);
    Array<float> histf  = cast<float, uint>(hist);
    Array<float> totals = createValueArray<float>(hDims, inDims[0] * inDims[1]);
    Array<float> weights =
        iota<float>(dim4(NUM_BINS), oDims);  // a.k.a histogram shape

    // pixel frequency probabilities
    auto freqs        = arithOp<float, af_div_t>(histf, totals, hDims);
    auto cumFreqs     = scan<af_add_t, float, float>(freqs, 0);
    auto oneMCumFreqs = arithOp<float, af_sub_t>(UnitP, cumFreqs, hDims);
    auto qLqH         = arithOp<float, af_mul_t>(cumFreqs, oneMCumFreqs, hDims);
    auto product      = arithOp<float, af_mul_t>(weights, freqs, hDims);
    auto cumProduct   = scan<af_add_t, float, float>(product, 0);
    auto weightedSum  = reduce<af_add_t, float, float>(product, 0);

    dim4 sigmaDims(NUM_BINS - 1, hDims[1], hDims[2], hDims[3]);
    Array<float> sigmas = createEmptyArray<float>(sigmaDims);
    for (unsigned b = 0; b < (NUM_BINS - 1); ++b) {
        const dim4 fDims(b + 1, hDims[1], hDims[2], hDims[3]);
        const dim4 eDims(NUM_BINS - 1 - b, hDims[1], hDims[2], hDims[3]);

        sliceIndex[0]    = {double(b), double(b), 1};
        seqBegin[0].end  = static_cast<double>(b);
        seqRest[0].begin = static_cast<double>(b + 1);

        auto qL    = createSubArray(cumFreqs, sliceIndex, false);
        auto qH    = arithOp<float, af_sub_t>(UnitP, qL, oDims);
        auto _muL  = createSubArray(cumProduct, sliceIndex, false);
        auto _muH  = arithOp<float, af_sub_t>(weightedSum, _muL, oDims);
        auto muL   = arithOp<float, af_div_t>(_muL, qL, oDims);
        auto muH   = arithOp<float, af_div_t>(_muH, qH, oDims);
        auto diff  = arithOp<float, af_sub_t>(muL, muH, oDims);
        auto sqrd  = arithOp<float, af_mul_t>(diff, diff, oDims);
        auto op2   = createSubArray(qLqH, sliceIndex, false);
        auto sigma = arithOp<float, af_mul_t>(sqrd, op2, oDims);

        auto binRes = createSubArray<float>(sigmas, sliceIndex, false);
        copyArray(binRes, sigma);
    }

    Array<float> thresh = createEmptyArray<float>(oDims);
    Array<uint> locs    = createEmptyArray<uint>(oDims);

    ireduce<af_max_t, float>(thresh, locs, sigmas, 0);

    return cast<float, uint>(
        arrayfire::common::tile(locs, dim4(inDims[0], inDims[1])));
}

Array<float> normalize(const Array<float>& supEdges, const float minVal,
                       const float maxVal) {
    auto minArray = createValueArray<float>(supEdges.dims(), minVal);
    auto diff  = arithOp<float, af_sub_t>(supEdges, minArray, supEdges.dims());
    auto denom = createValueArray<float>(supEdges.dims(), (maxVal - minVal));
    return arithOp<float, af_div_t>(diff, denom, supEdges.dims());
}

pair<Array<char>, Array<char>> computeCandidates(const Array<float>& supEdges,
                                                 const float t1,
                                                 const af_canny_threshold ct,
                                                 const float t2) {
    float maxVal =
        getScalar<float>(reduce_all<af_max_t, float, float>(supEdges));
    ;
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
            return make_pair(strong, weak);
        };
        default: {
            float minVal =
                getScalar<float>(reduce_all<af_min_t, float, float>(supEdges));
            auto normG = normalize(supEdges, minVal, maxVal);
            auto T2    = createValueArray<float>(supEdges.dims(), t2);
            auto T1    = createValueArray<float>(supEdges.dims(), t1);
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
    Array<float> cFilter = createHostDataArray<float>(dim4(5, 1), v.data());
    Array<float> rFilter = createHostDataArray<float>(dim4(1, 5), v.data());

    // Run separable convolution to smooth the input image
    Array<float> smt =
        convolve2<float, float>(cast<float, T>(in), cFilter, rFilter, false);

    auto g          = sobelDerivatives<float, float>(smt, sw);
    Array<float> gx = g.first;
    Array<float> gy = g.second;

    Array<float> gmag = gradientMagnitude(gx, gy, isf);

    Array<float> supEdges = nonMaximumSuppression(gmag, gx, gy);

    auto swpair = computeCandidates(supEdges, t1, ct, t2);

    return getHandle(edgeTrackingByHysteresis(swpair.first, swpair.second));
}

}  // namespace

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
