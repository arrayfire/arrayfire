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

    ASSERT_EQ(AF_SUCCESS, af_canny(&outArray, sArray, 0.24, 0.72, 3, true));

    char *outData = new char[sDims.elements()];

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    vector<char> currGoldBar = tests[0];
    size_t nElems        = currGoldBar.size();
    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])<< "at: " << elIter<< std::endl;
    }

    // cleanup
    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_release_array(sArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
}

TYPED_TEST(CannyEdgeDetector, ArraySizeLessThanBlockSize10x10)
{
    cannyTest<TypeParam>(string(TEST_DIR"/CannyEdgeDetector/fast10x10.test"));
}

TYPED_TEST(CannyEdgeDetector, ArraySizeEqualBlockSize16x16)
{
    cannyTest<TypeParam>(string(TEST_DIR"/CannyEdgeDetector/fast16x16.test"));
}

TEST(CannyEdgeDetector, InvalidSizeArray)
{
    af_array inArray   = 0;
    af_array outArray  = 0;

    vector<float>   in(100, 1);

    af::dim4 sDims(100, 1, 1, 1);

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &in.front(),
                sDims.ndims(), sDims.get(), (af_dtype) af::dtype_traits<float>::af_type));

    ASSERT_EQ(AF_ERR_SIZE, af_canny(&outArray, inArray, 0.24, 0.72, 3, true));

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

    ASSERT_EQ(AF_ERR_SIZE, af_canny(&outArray, inArray, 0.24, 0.72, 3, true));

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

    ASSERT_EQ(AF_ERR_ARG, af_canny(&outArray, inArray, 0.24, 0.72, 5, true));

    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
}
