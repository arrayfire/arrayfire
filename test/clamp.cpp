/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <af/array.h>
#include <af/arith.h>
#include <af/data.h>
#include <testHelpers.hpp>

using std::abs;
using namespace af;

const int num = 10000;

TEST(ClampTests, FloatArrayArray)
{
    af::array in = af::randu(num, f32);
    af::array lo = af::randu(num, f32)/10;       // Ensure lo <= 0.1
    af::array hi = 1.0 - af::randu(num, f32)/10; // Ensure hi >= 0.9
    af::eval(lo, hi);


    std::vector<float> hout(num), hin(num), hlo(num), hhi(num);
    af::array out = clamp(in, lo, hi);
    out.host(&hout[0]);
    in.host(&hin[0]);
    lo.host(&hlo[0]);
    hi.host(&hhi[0]);

    for (int i = 0; i < num; i++) {
        ASSERT_LE(hout[i], hhi[i]);
        ASSERT_GE(hout[i], hlo[i]);
        ASSERT_EQ(true, hout[i] == hin[i] || hout[i] == hlo[i] || hout[i] == hhi[i]);
    }
}

TEST(ClampTests, FloatArrayScalar)
{
    af::array in = af::randu(num, f32);
    af::array lo = af::randu(num, f32)/10; // Ensure lo <= 0.1
    float hi = 0.9;

    std::vector<float> hout(num), hin(num), hlo(num);
    af::array out = clamp(in, lo, hi);

    out.host(&hout[0]);
    in.host(&hin[0]);
    lo.host(&hlo[0]);

    for (int i = 0; i < num; i++) {
        ASSERT_LE(hout[i], hi);
        ASSERT_GE(hout[i], hlo[i]);
        ASSERT_EQ(true, hout[i] == hin[i] || hout[i] == hlo[i] || hout[i] == hi);
    }
}

TEST(ClampTests, FloatScalarArray)
{
    af::array in = af::randu(num, f32);
    float lo = 0.1;
    af::array hi = 1.0 - af::randu(num, f32)/10; // Ensure hi >= 0.9

    std::vector<float> hout(num), hin(num), hhi(num);
    af::array out = clamp(in, lo, hi);

    out.host(&hout[0]);
    in.host(&hin[0]);
    hi.host(&hhi[0]);

    for (int i = 0; i < num; i++) {
        ASSERT_LE(hout[i], hhi[i]);
        ASSERT_GE(hout[i], lo);
        ASSERT_EQ(true, hout[i] == hin[i] || hout[i] == lo || hout[i] == hhi[i]);
    }
}

TEST(ClampTests, FloatScalarScalar)
{
    af::array in = af::randu(num, f32);
    float lo = 0.1;
    float hi = 0.9;

    std::vector<float> hout(num), hin(num);
    af::array out = clamp(in, lo, hi);

    out.host(&hout[0]);
    in.host(&hin[0]);

    for (int i = 0; i < num; i++) {
        ASSERT_LE(hout[i], hi);
        ASSERT_GE(hout[i], lo);
        ASSERT_EQ(true, hout[i] == hin[i] || hout[i] == lo || hout[i] == hi);
    }
}
