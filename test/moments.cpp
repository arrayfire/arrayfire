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
    vector<af::dim4> numDims;

    vector<vector<T> >   in;
    vector<vector<float> >   tests;
    readTests<T, float, float>(pTestFile, numDims, in, tests);

    af::array imgArray(numDims.front(), &in.front()[0]);

    af::array momentsArray = af::moments(imgArray, AF_MOMENT_M00);
    T *mData = momentsArray.host<T>();
    for(int i=0; i<momentsArray.elements();++i) {
        ASSERT_NEAR(tests[0][i], mData[i], 4e-3 * tests[0][i] ) << "at: " << i << std::endl;
    }
    delete [] mData;

    momentsArray = af::moments(imgArray, AF_MOMENT_M01);
    mData = momentsArray.host<T>();
    for(int i=0; i<momentsArray.elements();++i) {
        ASSERT_NEAR(tests[1][i], mData[i], 8e-3 * tests[1][i] ) << "at: " << i << std::endl;
    }
    delete [] mData;

    momentsArray = af::moments(imgArray, AF_MOMENT_M10);
    mData = momentsArray.host<T>();
    for(int i=0; i<momentsArray.elements();++i) {
        ASSERT_NEAR(tests[2][i], mData[i], 3e-3 * tests[2][i] ) << "at: " << i << std::endl;
    }
    delete [] mData;

    momentsArray = af::moments(imgArray, AF_MOMENT_M11);
    mData = momentsArray.host<T>();
    for(int i=0; i<momentsArray.elements();++i) {
        ASSERT_NEAR(tests[3][i], mData[i], 7e-3 * tests[3][i] ) << "at: " << i << std::endl;
    }
    delete [] mData;
}

void momentsOnImageTest(string pTestFile, string pImageFile, bool isColor)
{
    vector<af::dim4> numDims;

    vector<vector<float> >   in;
    vector<vector<float> >   tests;
    readTests<float, float, float>(pTestFile, numDims, in, tests);

    af::array imgArray = af::loadImage(pImageFile.c_str(), isColor);

    double maxVal = af::max<double>(imgArray);
    double minVal = af::min<double>(imgArray);
    imgArray /= maxVal - minVal;

    af::array momentsArray = af::moments(imgArray, AF_MOMENT_M00);

    float *mData = momentsArray.host<float>();
    for(int i=0; i<momentsArray.elements();++i) {
        ASSERT_NEAR(tests[0][i], mData[i], 4e-3 * tests[0][i] ) << "at: " << i << std::endl;
    }
    delete [] mData;

    momentsArray = af::moments(imgArray, AF_MOMENT_M01);
    mData = momentsArray.host<float>();
    for(int i=0; i<momentsArray.elements();++i) {
        ASSERT_NEAR(tests[1][i], mData[i], 8e-3 * tests[1][i] ) << "at: " << i << std::endl;
    }
    delete [] mData;

    momentsArray = af::moments(imgArray, AF_MOMENT_M10);
    mData = momentsArray.host<float>();
    for(int i=0; i<momentsArray.elements();++i) {
        ASSERT_NEAR(tests[2][i], mData[i], 3e-3 * tests[2][i] ) << "at: " << i << std::endl;
    }
    delete [] mData;

    momentsArray = af::moments(imgArray, AF_MOMENT_M11);
    mData = momentsArray.host<float>();
    for(int i=0; i<momentsArray.elements();++i) {
        ASSERT_NEAR(tests[3][i], mData[i], 7e-3 * tests[3][i] ) << "at: " << i << std::endl;
    }
    delete [] mData;
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


