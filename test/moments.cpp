/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <arrayfire.h>
#include <af/array.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <testHelpers.hpp>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using af::cfloat;
using af::cdouble;

template<typename T>
class Image : public ::testing::Test
{
    public:
        virtual void SetUp() {
        }
};

// create a list of types to be tested
typedef ::testing::Types<float, double, int> TestTypes;

// register the type list
TYPED_TEST_CASE(Image, TestTypes);

template<typename T>
void momentsTest(string pTestFile)
{
    if (noDoubleTests<T>()) return;

    vector<af::dim4> numDims;

    vector<vector<T> >   in;
    vector<vector<float> >   tests;
    readTests<T, float, float>(pTestFile, numDims, in, tests);

    af::array imgArray(numDims.front(), &in.front()[0]);

    af::array momentsArray = af::moments(imgArray, AF_MOMENT_M00);
    vector<float> mData(momentsArray.elements());
    momentsArray.host(&mData[0]);
    for(int i=0; i<momentsArray.elements();++i) {
        ASSERT_NEAR(tests[0][i], mData[i], 4e-3 * tests[0][i] ) << "at: " << i << std::endl;
    }

    momentsArray = af::moments(imgArray, AF_MOMENT_M01);
    momentsArray.host(&mData[0]);
    for(int i=0; i<momentsArray.elements();++i) {
        ASSERT_NEAR(tests[1][i], mData[i], 8e-3 * tests[1][i] ) << "at: " << i << std::endl;
    }

    momentsArray = af::moments(imgArray, AF_MOMENT_M10);
    momentsArray.host(&mData[0]);
    for(int i=0; i<momentsArray.elements();++i) {
        ASSERT_NEAR(tests[2][i], mData[i], 3e-3 * tests[2][i] ) << "at: " << i << std::endl;
    }

    momentsArray = af::moments(imgArray, AF_MOMENT_M11);
    momentsArray.host(&mData[0]);
    for(int i=0; i<momentsArray.elements();++i) {
        ASSERT_NEAR(tests[3][i], mData[i], 7e-3 * tests[3][i] ) << "at: " << i << std::endl;
    }

    momentsArray = af::moments(imgArray, AF_MOMENT_FIRST_ORDER);
    mData.resize(momentsArray.elements());
    momentsArray.host(&mData[0]);
    for(int i=0; i<momentsArray.elements()/4;i+=4) {
        ASSERT_NEAR(tests[0][i], mData[i]  , 1e-3 * tests[0][i] ) << "at: " << i   << std::endl;
        ASSERT_NEAR(tests[1][i], mData[i+1], 1e-3 * tests[1][i] ) << "at: " << i+1 << std::endl;
        ASSERT_NEAR(tests[2][i], mData[i+2], 1e-3 * tests[2][i] ) << "at: " << i+2 << std::endl;
        ASSERT_NEAR(tests[3][i], mData[i+3], 1e-3 * tests[3][i] ) << "at: " << i+3 << std::endl;
    }
}

void momentsOnImageTest(string pTestFile, string pImageFile, bool isColor)
{
    if (noImageIOTests()) return;
    vector<af::dim4> numDims;

    vector<vector<float> >   in;
    vector<vector<float> >   tests;
    readTests<float, float, float>(pTestFile, numDims, in, tests);

    af::array imgArray = af::loadImage(pImageFile.c_str(), isColor);

    double maxVal = af::max<double>(imgArray);
    double minVal = af::min<double>(imgArray);
    imgArray -= minVal;
    imgArray /= maxVal - minVal;

    af::array momentsArray = af::moments(imgArray, AF_MOMENT_M00);

    vector<float> mData(momentsArray.elements());
    momentsArray.host(&mData[0]);
    for(int i=0; i<momentsArray.elements();++i) {
        ASSERT_NEAR(tests[0][i], mData[i], 1e-2 * tests[0][i] ) << "at: " << i << std::endl;
    }

    momentsArray = af::moments(imgArray, AF_MOMENT_M01);
    momentsArray.host(&mData[0]);
    for(int i=0; i<momentsArray.elements();++i) {
        ASSERT_NEAR(tests[1][i], mData[i], 1e-2 * tests[1][i] ) << "at: " << i << std::endl;
    }

    momentsArray = af::moments(imgArray, AF_MOMENT_M10);
    momentsArray.host(&mData[0]);
    for(int i=0; i<momentsArray.elements();++i) {
        ASSERT_NEAR(tests[2][i], mData[i], 1e-2 * tests[2][i] ) << "at: " << i << std::endl;
    }

    momentsArray = af::moments(imgArray, AF_MOMENT_M11);
    momentsArray.host(&mData[0]);
    for(int i=0; i<momentsArray.elements();++i) {
        ASSERT_NEAR(tests[3][i], mData[i], 1e-2 * tests[3][i] ) << "at: " << i << std::endl;
    }

    momentsArray = af::moments(imgArray, AF_MOMENT_FIRST_ORDER);
    mData.resize(momentsArray.elements());
    momentsArray.host(&mData[0]);
    for(int i=0; i<momentsArray.elements()/4;i+=4) {
        ASSERT_NEAR(tests[0][i], mData[i]  , 1e-2 * tests[0][i] ) << "at: " << i   << std::endl;
        ASSERT_NEAR(tests[1][i], mData[i+1], 1e-2 * tests[1][i] ) << "at: " << i+1 << std::endl;
        ASSERT_NEAR(tests[2][i], mData[i+2], 1e-2 * tests[2][i] ) << "at: " << i+2 << std::endl;
        ASSERT_NEAR(tests[3][i], mData[i+3], 1e-2 * tests[3][i] ) << "at: " << i+3 << std::endl;
    }
}

TEST(IMAGE, MomentsImage)
{
    momentsOnImageTest(string(TEST_DIR"/moments/gray_seq_16_moments.test"), string(TEST_DIR"/imageio/gray_seq_16.png"), false);
}

TEST(Image, MomentsImageBatch)
{
    momentsTest<float>(string(TEST_DIR"/moments/simple_mat_batch_moments.test"));
}

TEST(Image, MomentsBatch2D)
{
    momentsOnImageTest(string(TEST_DIR"/moments/color_seq_16_moments.test"), string(TEST_DIR"/imageio/color_seq_16.png"), true);
}

TYPED_TEST(Image, MomentsSynthTypes)
{
    momentsTest<TypeParam>(string(TEST_DIR"/moments/simple_mat_moments.test"));
}

TEST(Image, Moment_Issue1957)
{
    af::array A = af::identity(3, 3, b8);

    double m00;
    af::moments(&m00, A, AF_MOMENT_M00);
    ASSERT_EQ(m00, 3);
}
