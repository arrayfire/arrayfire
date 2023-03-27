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
#include <af/traits.hpp>
#include <cmath>
#include <string>
#include <vector>

using af::dim4;
using af::dtype_traits;
using std::abs;
using std::string;
using std::vector;

template<typename T, bool isColor>
void bilateralTest(string pTestFile) {
    SUPPORTED_TYPE_CHECK(T);
    IMAGEIO_ENABLED_CHECK();

    vector<dim4> inDims;
    vector<string> inFiles;
    vector<dim_t> outSizes;
    vector<string> outFiles;

    readImageTests(pTestFile, inDims, inFiles, outSizes, outFiles);

    size_t testCount = inDims.size();

    for (size_t testId = 0; testId < testCount; ++testId) {
        af_array inArray   = 0;
        af_array outArray  = 0;
        af_array goldArray = 0;
        dim_t nElems       = 0;

        inFiles[testId].insert(0, string(TEST_DIR "/bilateral/"));
        outFiles[testId].insert(0, string(TEST_DIR "/bilateral/"));

        ASSERT_SUCCESS(
            af_load_image(&inArray, inFiles[testId].c_str(), isColor));
        ASSERT_SUCCESS(
            af_load_image(&goldArray, outFiles[testId].c_str(), isColor));
        ASSERT_SUCCESS(af_get_elements(&nElems, goldArray));

        ASSERT_SUCCESS(
            af_bilateral(&outArray, inArray, 2.25f, 25.56f, isColor));

        ASSERT_IMAGES_NEAR(goldArray, outArray, 0.02f);

        ASSERT_SUCCESS(af_release_array(inArray));
        ASSERT_SUCCESS(af_release_array(outArray));
        ASSERT_SUCCESS(af_release_array(goldArray));
    }
}

TEST(BilateralOnImage, Grayscale) {
    bilateralTest<float, false>(string(TEST_DIR "/bilateral/gray.test"));
}

TEST(BilateralOnImage, Color) {
    bilateralTest<float, true>(string(TEST_DIR "/bilateral/color.test"));
}

template<typename T>
class BilateralOnData : public ::testing::Test {};

typedef ::testing::Types<float, double, int, uint, char, uchar, short, ushort>
    DataTestTypes;

// register the type list
TYPED_TEST_SUITE(BilateralOnData, DataTestTypes);

template<typename inType>
void bilateralDataTest(string pTestFile) {
    SUPPORTED_TYPE_CHECK(inType);

    typedef typename cond_type<is_same_type<inType, double>::value, double,
                               float>::type outType;

    vector<dim4> numDims;
    vector<vector<inType>> in;
    vector<vector<outType>> tests;

    readTests<inType, outType, float>(pTestFile, numDims, in, tests);

    dim4 dims         = numDims[0];
    af_array outArray = 0;
    af_array inArray  = 0;

    ASSERT_SUCCESS(af_create_array(&inArray, &(in[0].front()), dims.ndims(),
                                   dims.get(),
                                   (af_dtype)dtype_traits<inType>::af_type));

    ASSERT_SUCCESS(af_bilateral(&outArray, inArray, 2.25f, 25.56f, false));

    vector<outType> outData(dims.elements());

    ASSERT_SUCCESS(af_get_data_ptr((void*)outData.data(), outArray));

    for (size_t testIter = 0; testIter < tests.size(); ++testIter) {
        vector<outType> currGoldBar = tests[testIter];
        size_t nElems               = currGoldBar.size();
        ASSERT_EQ(true, compareArraysRMSD(nElems, &currGoldBar.front(),
                                          outData.data(), 0.02f));
    }

    // cleanup
    ASSERT_SUCCESS(af_release_array(inArray));
    ASSERT_SUCCESS(af_release_array(outArray));
}

TYPED_TEST(BilateralOnData, Rectangle) {
    bilateralDataTest<TypeParam>(string(TEST_DIR "/bilateral/rectangle.test"));
}

TYPED_TEST(BilateralOnData, Rectangle_Batch) {
    bilateralDataTest<TypeParam>(
        string(TEST_DIR "/bilateral/rectangle_batch.test"));
}

TYPED_TEST(BilateralOnData, InvalidArgs) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    vector<TypeParam> in(100, 1);

    af_array inArray  = 0;
    af_array outArray = 0;

    // check for color image bilateral
    dim4 dims = dim4(100, 1, 1, 1);
    ASSERT_SUCCESS(af_create_array(&inArray, &in.front(), dims.ndims(),
                                   dims.get(),
                                   (af_dtype)dtype_traits<TypeParam>::af_type));
    ASSERT_EQ(AF_ERR_SIZE,
              af_bilateral(&outArray, inArray, 0.12f, 0.34f, true));
    ASSERT_SUCCESS(af_release_array(inArray));
}

// C++ unit tests

using af::array;
using af::bilateral;

TEST(Bilateral, CPP) {
    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;

    readTests<float, float, float>(string(TEST_DIR "/bilateral/rectangle.test"),
                                   numDims, in, tests);

    dim4 dims = numDims[0];

    array a(dims, &(in[0].front()));
    array b = bilateral(a, 2.25f, 25.56f, false);

    vector<float> outData(dims.elements());
    b.host(outData.data());

    for (size_t testIter = 0; testIter < tests.size(); ++testIter) {
        vector<float> currGoldBar = tests[testIter];
        size_t nElems             = currGoldBar.size();
        ASSERT_EQ(true, compareArraysRMSD(nElems, currGoldBar.data(),
                                          outData.data(), 0.02f));
    }
}

using af::constant;
using af::iota;
using af::max;
using af::seq;
using af::span;

TEST(bilateral, GFOR) {
    dim4 dims = dim4(10, 10, 3);
    array A   = iota(dims);
    array B   = constant(0, dims);

    gfor(seq ii, 3) { B(span, span, ii) = bilateral(A(span, span, ii), 3, 5); }

    for (int ii = 0; ii < 3; ii++) {
        array c_ii = bilateral(A(span, span, ii), 3, 5);
        array b_ii = B(span, span, ii);
        ASSERT_EQ(max<double>(abs(c_ii - b_ii)) < 1E-5, true);
    }
}
