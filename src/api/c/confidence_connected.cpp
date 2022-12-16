/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/image.h>

#include <arith.hpp>
#include <common/cast.hpp>
#include <common/err_common.hpp>
#include <flood_fill.hpp>
#include <handle.hpp>
#include <imgproc_common.hpp>
#include <index.hpp>
#include <indexing_common.hpp>
#include <reduce.hpp>

#include <array>
#include <cmath>
#include <type_traits>

using af::dim4;
using arrayfire::common::cast;
using arrayfire::common::convRange;
using arrayfire::common::createSpanIndex;
using arrayfire::common::integralImage;
using detail::arithOp;
using detail::Array;
using detail::createValueArray;
using detail::reduce_all;
using detail::uchar;
using detail::uint;
using detail::ushort;
using std::conditional;
using std::is_same;
using std::sqrt;
using std::swap;

/// Index corner points of given seed points
template<typename T>
Array<T> pointList(const Array<T>& in, const Array<uint>& x,
                   const Array<uint>& y) {
    af_array xcoords                          = getHandle<uint>(x);
    af_array ycoords                          = getHandle<uint>(y);
    std::array<af_index_t, AF_MAX_DIMS> idxrs = {{{{xcoords}, false, false},
                                                  {{ycoords}, false, false},
                                                  createSpanIndex(),
                                                  createSpanIndex()}};

    Array<T> retVal = detail::index(in, idxrs.data());

    // detail::index fn keeps a reference to detail::Array
    // created from the xcoords/ycoords passed via idxrs.
    // Hence, it is safe to release xcoords, ycoords
    releaseHandle<uint>(xcoords);
    releaseHandle<uint>(ycoords);

    return retVal;
}

/// Returns the sum of all values given the four corner points of the region of
/// interest in the integral-image/summed-area-table of an input image.
///
///  +-------------------------------------+
///  |           |                |        |
///  |  A(_x, _y)|       B(_x, y_)|        |
///  |-----------+----------------+        |
///  |           |@@@@@@@@@@@@@@@@|        |
///  |           |@@@@@@@@@@@@@@@@|        |
///  |           |@@@@@@@@@@@@@@@@|        |
///  |           |@@@@@@@@@@@@@@@@|        |
///  |-----------+----------------+        |
///  |  C(x_, _y)        D(x_, y_)         |
///  |                                     |
///  +-------------------------------------+
template<typename T>
Array<T> sum(const Array<T>& sat, const Array<uint>& _x, const Array<uint>& x_,
             const Array<uint>& _y, const Array<uint>& y_) {
    Array<T> A  = pointList(sat, _x, _y);
    Array<T> B  = pointList(sat, _x, y_);
    Array<T> C  = pointList(sat, x_, _y);
    Array<T> D  = pointList(sat, x_, y_);
    Array<T> DA = arithOp<T, af_add_t>(D, A, D.dims());
    Array<T> BC = arithOp<T, af_add_t>(B, C, B.dims());
    return arithOp<T, af_sub_t>(DA, BC, DA.dims());
}

template<typename T>
af_array ccHelper(const Array<T>& img, const Array<uint>& seedx,
                  const Array<uint>& seedy, const unsigned radius,
                  const unsigned mult, const unsigned iterations,
                  const double segmentedValue) {
    using CT =
        typename conditional<is_same<T, double>::value, double, float>::type;
    constexpr CT epsilon = 1.0e-6;

    auto calcVar = [](CT s2, CT s1, CT n) -> CT {
        CT retVal = CT(0);
        if (n > 1) { retVal = (s2 - (s1 * s1 / n)) / (n - CT(1)); }
        return retVal;
    };

    const dim4& inDims       = img.dims();
    const dim4& seedDims     = seedx.dims();
    const size_t numSeeds    = seedx.elements();
    const unsigned nhoodLen  = 2 * radius + 1;
    const unsigned nhoodSize = nhoodLen * nhoodLen;

    auto labelSegmented = [segmentedValue, inDims](const Array<CT>& segmented) {
        Array<CT> newVals = createValueArray(inDims, CT(segmentedValue));
        Array<CT> result  = arithOp<CT, af_mul_t>(newVals, segmented, inDims);
        // cast final result to input type
        return cast<T, CT>(result);
    };

    Array<uint> radiip = createValueArray<uint>(seedDims, radius + 1);
    Array<uint> radii  = createValueArray<uint>(seedDims, radius);
    Array<uint> _x     = arithOp<uint, af_sub_t>(seedx, radiip, seedDims);
    Array<uint> x_     = arithOp<uint, af_add_t>(seedx, radii, seedDims);
    Array<uint> _y     = arithOp<uint, af_sub_t>(seedy, radiip, seedDims);
    Array<uint> y_     = arithOp<uint, af_add_t>(seedy, radii, seedDims);
    Array<CT> in       = convRange<CT, T>(img, CT(1), CT(2));
    Array<CT> in_2     = arithOp<CT, af_mul_t>(in, in, inDims);
    Array<CT> I1       = integralImage<CT>(in);
    Array<CT> I2       = integralImage<CT>(in_2);
    Array<CT> S1       = sum(I1, _x, x_, _y, y_);
    Array<CT> S2       = sum(I2, _x, x_, _y, y_);
    CT totSum          = reduce_all<af_add_t, CT, CT>(S1);
    CT totSumSq        = reduce_all<af_add_t, CT, CT>(S2);
    CT totalNum        = numSeeds * nhoodSize;
    CT s1mean          = totSum / totalNum;
    CT s1var           = calcVar(totSumSq, totSum, totalNum);
    CT s1stddev        = sqrt(s1var);
    CT lower           = s1mean - mult * s1stddev;
    CT upper           = s1mean + mult * s1stddev;

    Array<CT> seedIntensities = pointList(in, seedx, seedy);
    CT maxSeedIntensity       = reduce_all<af_max_t, CT, CT>(seedIntensities);
    CT minSeedIntensity       = reduce_all<af_min_t, CT, CT>(seedIntensities);

    if (lower > minSeedIntensity) { lower = minSeedIntensity; }
    if (upper < maxSeedIntensity) { upper = maxSeedIntensity; }

    Array<CT> segmented = floodFill(in, seedx, seedy, CT(1), lower, upper);

    if (std::abs<CT>(s1var) < epsilon) {
        // If variance is close to zero, stop after initial segmentation
        return getHandle(labelSegmented(segmented));
    }

    bool continueLoop = true;
    for (uint i = 0; (i < iterations) && continueLoop; ++i) {
        // Segmented images are set with 1's and 0's thus essentially
        // making them into mask arrays for each iteration's input image

        uint sampleCount = reduce_all<af_notzero_t, CT, uint>(segmented, true);
        if (sampleCount == 0) {
            // If no valid pixels are found, skip iterations
            break;
        }
        Array<CT> valids = arithOp<CT, af_mul_t>(segmented, in, inDims);
        Array<CT> vsqrd  = arithOp<CT, af_mul_t>(valids, valids, inDims);

        CT validsSum  = reduce_all<af_add_t, CT, CT>(valids, true);
        CT sumOfSqs   = reduce_all<af_add_t, CT, CT>(vsqrd, true);
        CT validsMean = validsSum / sampleCount;
        CT validsVar  = calcVar(sumOfSqs, validsSum, CT(sampleCount));
        CT stddev     = sqrt(validsVar);
        CT newLow     = validsMean - mult * stddev;
        CT newHigh    = validsMean + mult * stddev;

        if (newLow > minSeedIntensity) { newLow = minSeedIntensity; }
        if (newHigh < maxSeedIntensity) { newHigh = maxSeedIntensity; }

        if (std::abs<CT>(validsVar) < epsilon) {
            // If variance is close to zero, discontinue iterating.
            continueLoop = false;
        }
        segmented = floodFill(in, seedx, seedy, CT(1), newLow, newHigh);
    }

    return getHandle(labelSegmented(segmented));
}

af_err af_confidence_cc(af_array* out, const af_array in, const af_array seedx,
                        const af_array seedy, const unsigned radius,
                        const unsigned multiplier, const int iter,
                        const double segmented_value) {
    try {
        const ArrayInfo& inInfo         = getInfo(in);
        const ArrayInfo& seedxInfo      = getInfo(seedx);
        const ArrayInfo& seedyInfo      = getInfo(seedy);
        const af::dim4& inputDimensions = inInfo.dims();
        const af::dtype inputArrayType  = inInfo.getType();

        // TODO(pradeep) handle case where seeds are towards border
        //              and indexing may result in throwing exception
        // TODO(pradeep) add batch support later
        ARG_ASSERT(
            1, (inputDimensions.ndims() > 0 && inputDimensions.ndims() <= 2));

        ARG_ASSERT(2, (seedxInfo.ndims() == 1));
        ARG_ASSERT(3, (seedyInfo.ndims() == 1));
        ARG_ASSERT(2, (seedxInfo.elements() == seedyInfo.elements()));

        af_array output = 0;
        switch (inputArrayType) {
            case f32:
                output = ccHelper(getArray<float>(in), getArray<uint>(seedx),
                                  getArray<uint>(seedy), radius, multiplier,
                                  iter, segmented_value);
                break;
            case u32:
                output = ccHelper(getArray<uint>(in), getArray<uint>(seedx),
                                  getArray<uint>(seedy), radius, multiplier,
                                  iter, segmented_value);
                break;
            case u16:
                output = ccHelper(getArray<ushort>(in), getArray<uint>(seedx),
                                  getArray<uint>(seedy), radius, multiplier,
                                  iter, segmented_value);
                break;
            case u8:
                output = ccHelper(getArray<uchar>(in), getArray<uint>(seedx),
                                  getArray<uint>(seedy), radius, multiplier,
                                  iter, segmented_value);
                break;
            default: TYPE_ERROR(0, inputArrayType);
        }
        swap(*out, output);
    }
    CATCHALL;
    return AF_SUCCESS;
}
