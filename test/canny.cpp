/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <arrayfire.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <string>
#include <vector>
#include <testHelpers.hpp>

using std::string;
using std::vector;

template<typename T>
class CannyEdgeDetector : public ::testing::Test
{
    public:
        virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, int, uint, short, ushort, uchar, double> TestTypes;

// register the type list
TYPED_TEST_CASE(CannyEdgeDetector, TestTypes);

template<typename T>
void cannyTest(string pTestFile)
{
    if (noDoubleTests<T>()) return;

    vector<af::dim4>  numDims;
    vector<vector<T> >      in;
    vector<vector<char> >   tests;

    readTests<T, char, int>(pTestFile, numDims, in, tests);

    af::dim4 sDims    = numDims[0];
    af_array outArray = 0;
    af_array sArray   = 0;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&sArray, &(in[0].front()),
                sDims.ndims(), sDims.get(), (af_dtype)af::dtype_traits<T>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_canny(&outArray, sArray, AF_CANNY_THRESHOLD_MANUAL, 0.4147f, 0.8454f, 3, true));

    std::vector<char> outData(sDims.elements());

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData.data(), outArray));

    vector<char> currGoldBar = tests[0];
    size_t nElems        = currGoldBar.size();
    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])<< "at: " << elIter<< std::endl;
    }

    // cleanup
    ASSERT_EQ(AF_SUCCESS, af_release_array(sArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
}

TYPED_TEST(CannyEdgeDetector, ArraySizeLessThanBlockSize10x10)
{
    cannyTest<TypeParam>(string(TEST_DIR "/CannyEdgeDetector/fast10x10.test"));
}

TYPED_TEST(CannyEdgeDetector, ArraySizeEqualBlockSize16x16)
{
    cannyTest<TypeParam>(string(TEST_DIR "/CannyEdgeDetector/fast16x16.test"));
}

template<typename T>
void cannyImageOtsuTest(string pTestFile, bool isColor)
{
    if (noDoubleTests<T>()) return;
    if (noImageIOTests()) return;

    using af::dim4;

    vector<dim4>       inDims;
    vector<string>    inFiles;
    vector<dim_t>    outSizes;
    vector<string>   outFiles;

    readImageTests(pTestFile, inDims, inFiles, outSizes, outFiles);

    size_t testCount = inDims.size();

    for (size_t testId=0; testId<testCount; ++testId) {

        af_array inArray  = 0;
        af_array outArray = 0;
        af_array goldArray= 0;
        dim_t nElems   = 0;

        inFiles[testId].insert(0,string(TEST_DIR "/CannyEdgeDetector/"));
        outFiles[testId].insert(0,string(TEST_DIR "/CannyEdgeDetector/"));

        ASSERT_EQ(AF_SUCCESS, af_load_image(&inArray, inFiles[testId].c_str(), isColor));
        ASSERT_EQ(AF_SUCCESS, af_load_image_native(&goldArray, outFiles[testId].c_str()));
        ASSERT_EQ(AF_SUCCESS, af_get_elements(&nElems, goldArray));

        ASSERT_EQ(AF_SUCCESS, af_canny(&outArray, inArray, AF_CANNY_THRESHOLD_AUTO_OTSU, 0.08, 0.32, 3, false));

        std::vector<char> outData(nElems);
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData.data(), outArray));

        std::vector<char> goldData(nElems);
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)goldData.data(), goldArray));

        ASSERT_EQ(true, compareArraysRMSD(nElems, goldData.data(), outData.data(), 1.0e-3));

        ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(goldArray));
    }
}

TEST(CannyEdgeDetector, OtsuThreshold)
{
    cannyImageOtsuTest<float>(string(TEST_DIR "/CannyEdgeDetector/gray.test"), false);
}

TEST(CannyEdgeDetector, InvalidSizeArray)
{
    af_array inArray   = 0;
    af_array outArray  = 0;

    vector<float>   in(100, 1);

    af::dim4 sDims(100, 1, 1, 1);

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &in.front(),
                sDims.ndims(), sDims.get(), (af_dtype) af::dtype_traits<float>::af_type));

    ASSERT_EQ(AF_ERR_SIZE, af_canny(&outArray, inArray, AF_CANNY_THRESHOLD_MANUAL, 0.24, 0.72, 3, true));

    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
}

TEST(CannyEdgeDetector, Array4x4_Invalid)
{
    af_array inArray   = 0;
    af_array outArray  = 0;

    vector<float>   in(16, 1);

    af::dim4 sDims(4, 4, 1, 1);

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &in.front(),
                sDims.ndims(), sDims.get(), (af_dtype) af::dtype_traits<float>::af_type));

    ASSERT_EQ(AF_ERR_SIZE, af_canny(&outArray, inArray, AF_CANNY_THRESHOLD_MANUAL, 0.24, 0.72, 3, true));

    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
}

TEST(CannyEdgeDetector, Sobel5x5_Invalid)
{
    af_array inArray   = 0;
    af_array outArray  = 0;

    vector<float>   in(25, 1);

    af::dim4 sDims(5, 5, 1, 1);

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &in.front(),
                sDims.ndims(), sDims.get(), (af_dtype) af::dtype_traits<float>::af_type));

    ASSERT_EQ(AF_ERR_ARG, af_canny(&outArray, inArray, AF_CANNY_THRESHOLD_MANUAL, 0.24, 0.72, 5, true));

    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
}
