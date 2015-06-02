/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/image.h>
#include <af/index.h>
#include <af/data.h>

#include <ArrayInfo.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <arith.hpp>
#include <math.hpp>
#include <cast.hpp>
#include <tile.hpp>
#include <join.hpp>

using af::dim4;
using namespace detail;

template<typename T, typename cType>
static af_array rgb2gray(const af_array& in, const float r, const float g, const float b)
{
    Array<cType> input = cast<cType>(getArray<T>(in));
    dim4 inputDims = input.dims();
    dim4 matDims(inputDims[0], inputDims[1], 1, inputDims[3]);

    Array<cType> rCnst = createValueArray<cType>(matDims, scalar<cType>(r));
    Array<cType> gCnst = createValueArray<cType>(matDims, scalar<cType>(g));
    Array<cType> bCnst = createValueArray<cType>(matDims, scalar<cType>(b));

    // extract three channels as three slices
    af_seq slice1[4] = { af_span, af_span, {0, 0, 1}, af_span };
    af_seq slice2[4] = { af_span, af_span, {1, 1, 1}, af_span };
    af_seq slice3[4] = { af_span, af_span, {2, 2, 1}, af_span };

    af_array ch1Temp=0, ch2Temp=0, ch3Temp=0;
    AF_CHECK(af_index(&ch1Temp, in, 4, slice1));
    AF_CHECK(af_index(&ch2Temp, in, 4, slice2));
    AF_CHECK(af_index(&ch3Temp, in, 4, slice3));

    // r*Slice0
    Array<cType> expr1 = arithOp<cType, af_mul_t>(getArray<cType>(ch1Temp), rCnst, matDims);
    //g*Slice1
    Array<cType> expr2 = arithOp<cType, af_mul_t>(getArray<cType>(ch2Temp), gCnst, matDims);
    //b*Slice2
    Array<cType> expr3 = arithOp<cType, af_mul_t>(getArray<cType>(ch3Temp), bCnst, matDims);
    //r*Slice0 + g*Slice1
    Array<cType> expr4 = arithOp<cType, af_add_t>(expr1, expr2, matDims);
    //r*Slice0 + g*Slice1 + b*Slice2
    Array<cType> result= arithOp<cType, af_add_t>(expr3, expr4, matDims);

    AF_CHECK(af_release_array(ch1Temp));
    AF_CHECK(af_release_array(ch2Temp));
    AF_CHECK(af_release_array(ch3Temp));

    return getHandle<cType>(result);
}

template<typename T, typename cType>
static af_array gray2rgb(const af_array& in, const float r, const float g, const float b)
{
    if (r==1.0 && g==1.0 && b==1.0) {
        dim4 tileDims(1, 1, 3, 1);
        return getHandle(tile(getArray<T>(in), tileDims));
    }

    af_array mod_input = 0;
    dim4 inputDims = getInfo(in).dims();

    dim4 matDims(inputDims[0], inputDims[1], 1, inputDims[2]*inputDims[3]);

    AF_CHECK(af_moddims(&mod_input, in, matDims.ndims(), matDims.get()));
    Array<cType> mod_in = cast<cType>(getArray<cType>(mod_input));

    Array<cType> rCnst = createValueArray<cType>(matDims, scalar<cType>(r));
    Array<cType> gCnst = createValueArray<cType>(matDims, scalar<cType>(g));
    Array<cType> bCnst = createValueArray<cType>(matDims, scalar<cType>(b));

    Array<cType> expr1 = arithOp<cType, af_mul_t>(mod_in, rCnst, matDims);
    Array<cType> expr2 = arithOp<cType, af_mul_t>(mod_in, gCnst, matDims);
    Array<cType> expr3 = arithOp<cType, af_mul_t>(mod_in, bCnst, matDims);

    AF_CHECK(af_release_array(mod_input));

    // join channels
    Array<cType> expr4 = join<cType, cType>(2, expr1, expr2);
    return getHandle(join<cType, cType>(2, expr3, expr4));
}

template<typename T, typename cType, bool isRGB2GRAY>
static af_array convert(const af_array& in, const float r, const float g, const float b)
{
    if (isRGB2GRAY) {
        return rgb2gray<T, cType>(in, r, g, b);
    } else {
        return gray2rgb<T, cType>(in, r, g, b);
    }
}

template<bool isRGB2GRAY>
af_err convert(af_array* out, const af_array in, const float r, const float g, const float b)
{
    try {
        ArrayInfo info     = getInfo(in);
        af_dtype iType     = info.getType();
        af::dim4 inputDims = info.dims();

        ARG_ASSERT(1, (inputDims.ndims()>=2));
        if (isRGB2GRAY) ARG_ASSERT(1, (inputDims[2]==3));

        af_array output = 0;
        switch(iType) {
            case f64: output = convert<double, double, isRGB2GRAY>(in, r, g, b); break;
            case f32: output = convert<float , float , isRGB2GRAY>(in, r, g, b); break;
            case u32: output = convert<uint  , float , isRGB2GRAY>(in, r, g, b); break;
            case s32: output = convert<int   , float , isRGB2GRAY>(in, r, g, b); break;
            case u8:  output = convert<uchar , float , isRGB2GRAY>(in, r, g, b); break;
            default: TYPE_ERROR(1, iType); break;
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_rgb2gray(af_array* out, const af_array in, const float rPercent, const float gPercent, const float bPercent)
{
    return convert<true>(out, in, rPercent, gPercent, bPercent);
}

af_err af_gray2rgb(af_array* out, const af_array in, const float rFactor, const float gFactor, const float bFactor)
{
    return convert<false>(out, in, rFactor, gFactor, bFactor);
}
