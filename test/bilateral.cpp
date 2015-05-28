/*******************************************************
 * Copyright (c) 2014, ArrayFire
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
#include <cmath>
#include <testHelpers.hpp>

using std::string;
using std::vector;
using af::dim4;

template<typename T, bool isColor>
void bilateralTest(string pTestFile)
{
    if (noDoubleTests<T>()) return;

    vector<dim4>       inDims;
    vector<string>    inFiles;
    vector<dim_t> outSizes;
    vector<string>   outFiles;

    readImageTests(pTestFile, inDims, inFiles, outSizes, outFiles);

    size_t testCount = inDims.size();

    for (size_t testId=0; testId<testCount; ++testId) {

        af_array inArray  = 0;
        af_array outArray = 0;
        af_array goldArray= 0;
        dim_t nElems   = 0;

        inFiles[testId].insert(0,string(TEST_DIR"/bilateral/"));
        outFiles[testId].insert(0,string(TEST_DIR"/bilateral/"));

        ASSERT_EQ(AF_SUCCESS, af_load_image(&inArray, inFiles[testId].c_str(), isColor));
        ASSERT_EQ(AF_SUCCESS, af_load_image(&goldArray, outFiles[testId].c_str(), isColor));
        ASSERT_EQ(AF_SUCCESS, af_get_elements(&nElems, goldArray));

        ASSERT_EQ(AF_SUCCESS, af_bilateral(&outArray, inArray, 2.25f, 25.56f, isColor));

        T * outData = new T[nElems];
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

        T * goldData= new T[nElems];
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)goldData, goldArray));

        ASSERT_EQ(true, compareArraysRMSD(nElems, goldData, outData, 0.02f));

        ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(goldArray));
    }
}

TEST(BilateralOnImage, Grayscale)
{
    bilateralTest<float, false>(string(TEST_DIR"/bilateral/gray.test"));
}

TEST(BilateralOnImage, Color)
{
    bilateralTest<float, true>(string(TEST_DIR"/bilateral/color.test"));
}


template<typename T>
class BilateralOnData : public ::testing::Test
{
};

typedef ::testing::Types<float, double, int, uint, char, uchar> DataTestTypes;

// register the type list
TYPED_TEST_CASE(BilateralOnData, DataTestTypes);

template<typename inType>
void bilateralDataTest(string pTestFile)
{
    if (noDoubleTests<inType>()) return;

    typedef typename cond_type<is_same_type<inType, double>::value, double, float>::type outType;

    vector<af::dim4>        numDims;
    vector<vector<inType> >       in;
    vector<vector<outType> >   tests;

    readTests<inType, outType, float>(pTestFile, numDims, in, tests);

    af::dim4 dims      = numDims[0];
    af_array outArray  = 0;
    af_array inArray   = 0;
    outType *outData;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in[0].front()),
                dims.ndims(), dims.get(), (af_dtype)af::dtype_traits<inType>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_bilateral(&outArray, inArray, 2.25f, 25.56f, false));

    outData = new outType[dims.elements()];

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    for (size_t testIter=0; testIter<tests.size(); ++testIter) {
        vector<outType> currGoldBar = tests[testIter];
        size_t nElems = currGoldBar.size();
        ASSERT_EQ(true, compareArraysRMSD(nElems, &currGoldBar.front(), outData, 0.02f));
    }

    // cleanup
    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
}

TYPED_TEST(BilateralOnData, Rectangle)
{
    bilateralDataTest<TypeParam>(string(TEST_DIR"/bilateral/rectangle.test"));
}

TYPED_TEST(BilateralOnData, Rectangle_Batch)
{
    bilateralDataTest<TypeParam>(string(TEST_DIR"/bilateral/rectangle_batch.test"));
}

TYPED_TEST(BilateralOnData, InvalidArgs)
{
    if (noDoubleTests<TypeParam>()) return;

    vector<TypeParam>   in(100,1);

    af_array inArray   = 0;
    af_array outArray  = 0;

    // check for color image bilateral
    af::dim4 dims = af::dim4(100,1,1,1);
    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &in.front(),
                dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<TypeParam>::af_type));
    ASSERT_EQ(AF_ERR_SIZE, af_bilateral(&outArray, inArray, 0.12f, 0.34f, true));
    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
}

// C++ unit tests
TEST(Bilateral, CPP)
{
    if (noDoubleTests<float>()) return;

    using af::array;

    vector<af::dim4>      numDims;
    vector<vector<float> >      in;
    vector<vector<float> >   tests;

    readTests<float, float, float>(string(TEST_DIR"/bilateral/rectangle.test"), numDims, in, tests);

    af::dim4 dims      = numDims[0];

    array a(dims, &(in[0].front()));
    array b = af::bilateral(a, 2.25f, 25.56f, false);

    float *outData = new float[dims.elements()];
    b.host(outData);

    for (size_t testIter=0; testIter<tests.size(); ++testIter) {
        vector<float> currGoldBar = tests[testIter];
        size_t nElems = currGoldBar.size();
        ASSERT_EQ(true, compareArraysRMSD(nElems, &currGoldBar.front(), outData, 0.02f));
    }

    // cleanup
    delete[] outData;
}


TEST(bilateral, GFOR)
{
    using namespace af;

    dim4 dims = dim4(10, 10, 3);
    array A = iota(dims);
    array B = constant(0, dims);

    gfor(seq ii, 3) {
        B(span, span, ii) = bilateral(A(span, span, ii), 3, 5);
    }

    for(int ii = 0; ii < 3; ii++) {
        array c_ii = bilateral(A(span, span, ii), 3, 5);
        array b_ii = B(span, span, ii);
        ASSERT_EQ(max<double>(abs(c_ii - b_ii)) < 1E-5, true);
    }
}
