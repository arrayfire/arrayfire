/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/defines.h>
#include <af/image.h>
#include <handle.hpp>
#include <err_common.hpp>
#include <backend.hpp>
#include <Array.hpp>
#include <arith.hpp>
#include <canny.hpp>
#include <complex.hpp>
#include <convolve.hpp>
#include <copy.hpp>
#include <histogram.hpp>
#include <logic.hpp>
#include <sobel.hpp>
#include <transpose.hpp>
#include <unary.hpp>
#include <utility>

using af::dim4;
using namespace detail;

Array<float> gradientMagnitude(const Array<float>& gx, const Array<float>& gy, const bool& isf)
{
    if (isf) {
        Array<float> gx2 = detail::abs<float, float>(gx);
        Array<float> gy2 = detail::abs<float, float>(gy);
        return detail::arithOp<float, af_add_t>(gx2, gy2, gx2.dims());
    } else {
        Array<float> gx2 = detail::arithOp<float, af_mul_t>(gx, gx, gx.dims());
        Array<float> gy2 = detail::arithOp<float, af_mul_t>(gy, gy, gy.dims());
        Array<float> sg  = detail::arithOp<float, af_add_t>(gx2, gy2, gx2.dims());
        return detail::unaryOp<float, af_sqrt_t>(sg);
    }
}

template<typename T>
af_array cannyHelper(const Array<T> in, const float t1, const float t2,
                     const unsigned sw, const bool isf)
{
    typedef typename std::pair< Array<float>, Array<float> > GradientPair;

    static const std::vector<float> v = {-0.11021, -0.23691, -0.30576, -0.23691, -0.11021};
    Array<float> cFilter= detail::createHostDataArray<float>(dim4(5, 1), v.data());
    Array<float> rFilter= detail::createHostDataArray<float>(dim4(1, 5), v.data());

    // Run separable convolution to smooth the input image
    Array<float> smt = detail::convolve2<float, float, false>(cast<float, T>(in), cFilter, rFilter);

    GradientPair g  = detail::sobelDerivatives<float, float>(smt, sw);
    Array<float> gx = g.first;
    Array<float> gy = g.second;

    Array<float> gmag = gradientMagnitude(gx, gy, isf);

    Array<float> supEdges = detail::nonMaximumSuppression(gmag, gx, gy);

    const af::dim4 dims = in.dims();

    int width    = dims[0];
    int height   = dims[1];
    int channels = dims[2];

    std::vector<uint> hostHist(256*channels);

    Array<uint> hist = detail::histogram<float, uint, false>(supEdges, 256, 0, 255);

    detail::copyData(hostHist.data(), hist);

    Array<float> T2 = detail::createEmptyArray<float>(gmag.dims());
    Array<float> T1 = detail::createEmptyArray<float>(gmag.dims());

    for (dim_t c=0; c<channels; ++c)
    {
        std::vector<af_seq> sliceIndex(4, af_span);

        sliceIndex[2] = {double(c), double(c), 1};

        dim_t offset = c*256;

        // find the number of pixels where tHigh(percentage) of total pixels except zeros
        int highCount = (int)((width*height - hostHist[offset]) * t2 + 0.5f);

        // compute high level trigger value using histogram distribution
        // i is set to max value in unsigned char
        int i = 255;
        for (int sum=width*height-hostHist[offset]; sum>highCount; sum-=hostHist[i--]);

        float highValue = (float)++i;
        float lowValue  = (float)(highValue * t1 + 0.5f);

        Array<float> highTslice = createSubArray<float>(T2, sliceIndex, false);
        Array<float> lowTslice  = createSubArray<float>(T1, sliceIndex, false);

        copyArray<float>(highTslice, createValueArray<float>(af::dim4(width, height), highValue));
        copyArray<float>( lowTslice, createValueArray<float>(af::dim4(width, height),  lowValue));
    }

    Array<char> weak1  = detail::logicOp<float, af_ge_t >(supEdges,    T1, supEdges.dims());
    Array<char> weak2  = detail::logicOp<float, af_lt_t >(supEdges,    T2, supEdges.dims());
    Array<char> weak   = detail::logicOp<char , af_and_t>(   weak1, weak2,    weak1.dims());
    Array<char> strong = detail::logicOp<float, af_ge_t >(supEdges,    T2, supEdges.dims());

    return getHandle(detail::edgeTrackingByHysteresis(strong, weak));
}

af_err af_canny(af_array* out, const af_array in, const float t1, const float t2,
                const unsigned sw, const bool isf)
{
    try {
        const ArrayInfo& info = getInfo(in);
        af::dim4 dims  = info.dims();

        DIM_ASSERT(2, (dims.ndims() >= 2));
        // Input should be a minimum of 5x5 image
        // since the gaussian filter used for smoothing
        // the input is of 5x5 size. It's not mandatory but
        // it is essentially no use if image is less than 5x5
        DIM_ASSERT(2, (dims[0]>=5 && dims[1]>=5));
        ARG_ASSERT(5, (sw==3));

        af_array output;

        af_dtype type  = info.getType();
        switch(type) {
            case f32: output = cannyHelper<float >(getArray<float >(in), t1, t2, sw, isf); break;
            case f64: output = cannyHelper<double>(getArray<double>(in), t1, t2, sw, isf); break;
            case s32: output = cannyHelper<int   >(getArray<int   >(in), t1, t2, sw, isf); break;
            case u32: output = cannyHelper<uint  >(getArray<uint  >(in), t1, t2, sw, isf); break;
            case s16: output = cannyHelper<short >(getArray<short >(in), t1, t2, sw, isf); break;
            case u16: output = cannyHelper<ushort>(getArray<ushort>(in), t1, t2, sw, isf); break;
            case u8:  output = cannyHelper<uchar >(getArray<uchar >(in), t1, t2, sw, isf); break;
            default : TYPE_ERROR(1, type);
        }
        // output array is binary array
        std::swap(output, *out);
    }
    CATCHALL;

    return AF_SUCCESS;
}
