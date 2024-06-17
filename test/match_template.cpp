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
#include <string>
#include <vector>

using af::array;
using af::dim4;
using af::dtype_traits;
using af::exception;
using std::cout;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class MatchTemplate : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double, int, uint, char, uchar, short, ushort>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(MatchTemplate, TestTypes);

template<typename T>
void matchTemplateTest(string pTestFile, af_match_type pMatchType) {
    typedef
        typename cond_type<is_same_type<T, double>::value, double, float>::type
            outType;
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;
    vector<vector<T>> in;
    vector<vector<outType>> tests;

    readTests<T, outType, float>(pTestFile, numDims, in, tests);

    dim4 sDims        = numDims[0];
    dim4 tDims        = numDims[1];
    af_array outArray = 0;
    af_array sArray   = 0;
    af_array tArray   = 0;

    ASSERT_SUCCESS(af_create_array(&sArray, &(in[0].front()), sDims.ndims(),
                                   sDims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    ASSERT_SUCCESS(af_create_array(&tArray, &(in[1].front()), tDims.ndims(),
                                   tDims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    ASSERT_SUCCESS_CHECK_SUPRT(af_match_template(&outArray, sArray, tArray, pMatchType));

    vector<outType> outData(sDims.elements());

    ASSERT_SUCCESS(af_get_data_ptr((void *)outData.data(), outArray));

    vector<outType> currGoldBar = tests[0];
    size_t nElems               = currGoldBar.size();
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_NEAR(currGoldBar[elIter], outData[elIter], 1.0e-3)
            << "at: " << elIter << endl;
    }

    // cleanup
    ASSERT_SUCCESS(af_release_array(sArray));
    ASSERT_SUCCESS(af_release_array(tArray));
    ASSERT_SUCCESS(af_release_array(outArray));
}

TYPED_TEST(MatchTemplate, Matrix_SAD) {
    matchTemplateTest<TypeParam>(
        string(TEST_DIR "/MatchTemplate/matrix_sad.test"), AF_SAD);
}

TYPED_TEST(MatchTemplate, Matrix_SSD) {
    matchTemplateTest<TypeParam>(
        string(TEST_DIR "/MatchTemplate/matrix_ssd.test"), AF_SSD);
}

TYPED_TEST(MatchTemplate, MatrixBatch_SAD) {
    matchTemplateTest<TypeParam>(
        string(TEST_DIR "/MatchTemplate/matrix_sad_batch.test"), AF_SAD);
}

TEST(MatchTemplate, InvalidMatchType) {
    af_array inArray  = 0;
    af_array tArray   = 0;
    af_array outArray = 0;

    vector<float> in(100, 1);

    dim4 sDims(10, 10, 1, 1);
    dim4 tDims(4, 4, 1, 1);

    ASSERT_SUCCESS(af_create_array(&inArray, &in.front(), sDims.ndims(),
                                   sDims.get(),
                                   (af_dtype)dtype_traits<float>::af_type));

    ASSERT_SUCCESS(af_create_array(&tArray, &in.front(), tDims.ndims(),
                                   tDims.get(),
                                   (af_dtype)dtype_traits<float>::af_type));

    ASSERT_EQ(AF_ERR_ARG,
              af_match_template(&outArray, inArray, tArray, (af_match_type)-1));

    ASSERT_SUCCESS(af_release_array(inArray));
    ASSERT_SUCCESS(af_release_array(tArray));
}

///////////////////////////////// CPP TESTS /////////////////////////////
//
TEST(MatchTemplate, CPP) {
    vector<float> in(100, 1);

    dim4 sDims(10, 10, 1, 1);
    dim4 tDims(4, 4, 1, 1);

    try {
        array input(sDims, &in.front());
        array tmplt(tDims, &in.front());

        array out = matchTemplate(input, tmplt, (af_match_type)-1);
    } catch (exception &e) {
        cout << "Invalid Match test: " << e.what() << endl;
    }
}
