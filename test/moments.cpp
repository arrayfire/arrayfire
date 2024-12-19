/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <af/array.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <iostream>
#include <string>
#include <vector>

using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using af::identity;
using af::loadImage;
using af::max;
using af::min;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Image : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double, int> TestTypes;

// register the type list
TYPED_TEST_SUITE(Image, TestTypes);

template<typename T>
void momentsTest(string pTestFile) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;

    vector<vector<T>> in;
    vector<vector<float>> tests;
    readTests<T, float, float>(pTestFile, numDims, in, tests);

    array imgArray(numDims.front(), &in.front()[0]);

    array momentsArray;
    try { momentsArray = moments(imgArray, AF_MOMENT_M00); } catch FUNCTION_UNSUPPORTED
    vector<float> mData(momentsArray.elements());
    momentsArray.host(&mData[0]);
    for (int i = 0; i < momentsArray.elements(); ++i) {
        ASSERT_NEAR(tests[0][i], mData[i], 4e-3 * tests[0][i])
            << "at: " << i << endl;
    }

    try { momentsArray = moments(imgArray, AF_MOMENT_M01); } catch FUNCTION_UNSUPPORTED
    momentsArray.host(&mData[0]);
    for (int i = 0; i < momentsArray.elements(); ++i) {
        ASSERT_NEAR(tests[1][i], mData[i], 8e-3 * tests[1][i])
            << "at: " << i << endl;
    }

    try { momentsArray = moments(imgArray, AF_MOMENT_M10); } catch FUNCTION_UNSUPPORTED
    momentsArray.host(&mData[0]);
    for (int i = 0; i < momentsArray.elements(); ++i) {
        ASSERT_NEAR(tests[2][i], mData[i], 3e-3 * tests[2][i])
            << "at: " << i << endl;
    }

    try { momentsArray = moments(imgArray, AF_MOMENT_M11); } catch FUNCTION_UNSUPPORTED
    momentsArray.host(&mData[0]);
    for (int i = 0; i < momentsArray.elements(); ++i) {
        ASSERT_NEAR(tests[3][i], mData[i], 7e-3 * tests[3][i])
            << "at: " << i << endl;
    }

    try { momentsArray = moments(imgArray, AF_MOMENT_FIRST_ORDER); } catch FUNCTION_UNSUPPORTED
    mData.resize(momentsArray.elements());
    momentsArray.host(&mData[0]);
    for (int i = 0; i < momentsArray.elements() / 4; i += 4) {
        ASSERT_NEAR(tests[0][i], mData[i], 1e-3 * tests[0][i])
            << "at: " << i << endl;
        ASSERT_NEAR(tests[1][i], mData[i + 1], 1e-3 * tests[1][i])
            << "at: " << i + 1 << endl;
        ASSERT_NEAR(tests[2][i], mData[i + 2], 1e-3 * tests[2][i])
            << "at: " << i + 2 << endl;
        ASSERT_NEAR(tests[3][i], mData[i + 3], 1e-3 * tests[3][i])
            << "at: " << i + 3 << endl;
    }
}

void momentsOnImageTest(string pTestFile, string pImageFile, bool isColor) {
    IMAGEIO_ENABLED_CHECK();
    vector<dim4> numDims;

    vector<vector<float>> in;
    vector<vector<float>> tests;
    readTests<float, float, float>(pTestFile, numDims, in, tests);

    array imgArray = loadImage(pImageFile.c_str(), isColor);

    double maxVal = max<double>(imgArray);
    double minVal = min<double>(imgArray);
    imgArray -= minVal;
    imgArray /= maxVal - minVal;

    array momentsArray;
    try { momentsArray = moments(imgArray, AF_MOMENT_M00); } catch FUNCTION_UNSUPPORTED

    vector<float> mData(momentsArray.elements());
    momentsArray.host(&mData[0]);
    for (int i = 0; i < momentsArray.elements(); ++i) {
        ASSERT_NEAR(tests[0][i], mData[i], 1e-2 * tests[0][i])
            << "at: " << i << endl;
    }

    try { momentsArray = moments(imgArray, AF_MOMENT_M01); } catch FUNCTION_UNSUPPORTED
    momentsArray.host(&mData[0]);
    for (int i = 0; i < momentsArray.elements(); ++i) {
        ASSERT_NEAR(tests[1][i], mData[i], 1e-2 * tests[1][i])
            << "at: " << i << endl;
    }

    try { momentsArray = moments(imgArray, AF_MOMENT_M10); } catch FUNCTION_UNSUPPORTED
    momentsArray.host(&mData[0]);
    for (int i = 0; i < momentsArray.elements(); ++i) {
        ASSERT_NEAR(tests[2][i], mData[i], 1e-2 * tests[2][i])
            << "at: " << i << endl;
    }

    try { momentsArray = moments(imgArray, AF_MOMENT_M11); } catch FUNCTION_UNSUPPORTED
    momentsArray.host(&mData[0]);
    for (int i = 0; i < momentsArray.elements(); ++i) {
        ASSERT_NEAR(tests[3][i], mData[i], 1e-2 * tests[3][i])
            << "at: " << i << endl;
    }

    try { momentsArray = moments(imgArray, AF_MOMENT_FIRST_ORDER); } catch FUNCTION_UNSUPPORTED
    mData.resize(momentsArray.elements());
    momentsArray.host(&mData[0]);
    for (int i = 0; i < momentsArray.elements() / 4; i += 4) {
        ASSERT_NEAR(tests[0][i], mData[i], 1e-2 * tests[0][i])
            << "at: " << i << endl;
        ASSERT_NEAR(tests[1][i], mData[i + 1], 1e-2 * tests[1][i])
            << "at: " << i + 1 << endl;
        ASSERT_NEAR(tests[2][i], mData[i + 2], 1e-2 * tests[2][i])
            << "at: " << i + 2 << endl;
        ASSERT_NEAR(tests[3][i], mData[i + 3], 1e-2 * tests[3][i])
            << "at: " << i + 3 << endl;
    }
}

TEST(IMAGE, MomentsImage) {
    momentsOnImageTest(string(TEST_DIR "/moments/gray_seq_16_moments.test"),
                       string(TEST_DIR "/imageio/gray_seq_16.png"), false);
}

TEST(Image, MomentsImageBatch) {
    momentsTest<float>(
        string(TEST_DIR "/moments/simple_mat_batch_moments.test"));
}

TEST(Image, MomentsBatch2D) {
    momentsOnImageTest(string(TEST_DIR "/moments/color_seq_16_moments.test"),
                       string(TEST_DIR "/imageio/color_seq_16.png"), true);
}

TYPED_TEST(Image, MomentsSynthTypes) {
    momentsTest<TypeParam>(string(TEST_DIR "/moments/simple_mat_moments.test"));
}

TEST(Image, Moment_Issue1957) {
    array A = identity(3, 3, b8);

    double m00;
    try { moments(&m00, A, AF_MOMENT_M00); } catch FUNCTION_UNSUPPORTED
    ASSERT_EQ(m00, 3);
}
