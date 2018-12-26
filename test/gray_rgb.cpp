/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <af/dim4.hpp>
#include <string>
#include <vector>

using af::array;
using af::randu;
using std::vector;

TEST(rgb_gray, 32bit) {
    array rgb  = randu(10, 10, 3);
    array gray = rgb2gray(rgb);

    vector<float> h_rgb(rgb.elements());
    vector<float> h_gray(gray.elements());

    rgb.host(&h_rgb[0]);
    gray.host(&h_gray[0]);

    int num  = gray.elements();
    int roff = 0;
    int goff = num;
    int boff = 2 * num;

    const float rPercent = 0.2126f;
    const float gPercent = 0.7152f;
    const float bPercent = 0.0722f;

    for (int i = 0; i < num; i++) {
        float res = rPercent * h_rgb[i + roff] + gPercent * h_rgb[i + goff] +
                    bPercent * h_rgb[i + boff];

        ASSERT_FLOAT_EQ(res, h_gray[i]);
    }
}

TEST(rgb_gray, 8bit) {
    array rgb  = randu(10, 10, 3, u8);
    array gray = rgb2gray(rgb);

    vector<uchar> h_rgb(rgb.elements());
    vector<float> h_gray(gray.elements());

    rgb.host(&h_rgb[0]);
    gray.host(&h_gray[0]);

    int num  = gray.elements();
    int roff = 0;
    int goff = num;
    int boff = 2 * num;

    const float rPercent = 0.2126f;
    const float gPercent = 0.7152f;
    const float bPercent = 0.0722f;

    for (int i = 0; i < num; i++) {
        float res = rPercent * h_rgb[i + roff] + gPercent * h_rgb[i + goff] +
                    bPercent * h_rgb[i + boff];

        ASSERT_FLOAT_EQ(res, h_gray[i]);
    }
}

TEST(gray_rgb, 32bit) {
    array gray = randu(10, 10);

    const float rPercent = 0.33f;
    const float gPercent = 0.34f;
    const float bPercent = 0.33f;

    array rgb = gray2rgb(gray, rPercent, gPercent, bPercent);
    vector<float> h_rgb(rgb.elements());
    vector<float> h_gray(gray.elements());

    int num  = gray.elements();
    int roff = 0;
    int goff = num;
    int boff = 2 * num;

    for (int i = 0; i < num; i++) {
        float gray = h_gray[i];

        float r = rPercent * gray;
        float g = gPercent * gray;
        float b = bPercent * gray;

        ASSERT_FLOAT_EQ(r, h_rgb[i + roff]);
        ASSERT_FLOAT_EQ(g, h_rgb[i + goff]);
        ASSERT_FLOAT_EQ(b, h_rgb[i + boff]);
    }
}

TEST(rgb_gray, MaxDim) {
    size_t largeDim = 65535 * 32 + 1;
    array rgb       = randu(1, largeDim, 3, u8);
    array gray      = rgb2gray(rgb);

    vector<uchar> h_rgb(rgb.elements());
    vector<float> h_gray(gray.elements());

    rgb.host(&h_rgb[0]);
    gray.host(&h_gray[0]);

    int num  = gray.elements();
    int roff = 0;
    int goff = num;
    int boff = 2 * num;

    const float rPercent = 0.2126f;
    const float gPercent = 0.7152f;
    const float bPercent = 0.0722f;

    for (int i = 0; i < num; i++) {
        float res = rPercent * h_rgb[i + roff] + gPercent * h_rgb[i + goff] +
                    bPercent * h_rgb[i + boff];

        ASSERT_FLOAT_EQ(res, h_gray[i]);
    }
}
