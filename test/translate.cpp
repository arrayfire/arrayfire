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
class Translate : public ::testing::Test
{
    public:
        virtual void SetUp() {
        }
};

template<typename T>
class TranslateInt : public ::testing::Test
{
    public:
        virtual void SetUp() {
        }
};

// create a list of types to be tested
typedef ::testing::Types<float, double, cfloat, cdouble> TestTypes;
typedef ::testing::Types<int, intl, char> TestTypesInt;

// register the type list
TYPED_TEST_CASE(Translate, TestTypes);
TYPED_TEST_CASE(TranslateInt, TestTypesInt);

template<typename T>
void translateTest(string pTestFile, const unsigned resultIdx, af::dim4 odims, const float tx, const float ty, const af_interp_type method, const float max_fail_count = 0.0001)
{
    if (noDoubleTests<T>()) return;

    vector<af::dim4> numDims;
    vector<vector<T> >   in;
    vector<vector<float> >   tests;
    readTests<T, float, float>(pTestFile,numDims,in,tests);

    af_array inArray = 0;
    af_array outArray = 0;

    af::dim4 dims = numDims[0];

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in[0].front()), dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<T>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_translate(&outArray, inArray, tx, ty, odims[0], odims[1], method));

    // Get result
    T* outData = new T[tests[resultIdx].size()];
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    // Compare result
    size_t nElems = tests[resultIdx].size();

    size_t fail_count = 0;
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        if(abs((T)tests[resultIdx][elIter] - outData[elIter]) > 0.0001) {
            fail_count++;
        }
    }
    ASSERT_EQ(true, (((float)fail_count / (float)(nElems)) <= max_fail_count))
             << "Fail Count  = " << fail_count << std::endl;

    // Delete
    delete[] outData;

    if(inArray   != 0) af_release_array(inArray);
    if(outArray  != 0) af_release_array(outArray);
}

TYPED_TEST(Translate, Small1)
{
    translateTest<TypeParam>(string(TEST_DIR"/translate/translate_small_1.test"), 0,
                             af::dim4(10, 10, 1, 1), 3, 2, AF_INTERP_NEAREST);
}

TYPED_TEST(Translate, Small2)
{
    translateTest<TypeParam>(string(TEST_DIR"/translate/translate_small_1.test"), 1,
                             af::dim4(10, 10, 1, 1), -3, -2, AF_INTERP_NEAREST);
}

TYPED_TEST(Translate, Small3)
{
    translateTest<TypeParam>(string(TEST_DIR"/translate/translate_small_1.test"), 2,
                             af::dim4(15, 15, 1, 1), 1.5, 2.5, AF_INTERP_BILINEAR);
}

TYPED_TEST(Translate, Small4)
{
    translateTest<TypeParam>(string(TEST_DIR"/translate/translate_small_1.test"), 3,
                             af::dim4(15, 15, 1, 1), -1.5, -2.5, AF_INTERP_BILINEAR);
}

TYPED_TEST(Translate, Large1)
{
    translateTest<TypeParam>(string(TEST_DIR"/translate/translate_large_1.test"), 0,
                             af::dim4(250, 320, 1, 1), 10, 18, AF_INTERP_NEAREST);
}

TYPED_TEST(Translate, Large2)
{
    translateTest<TypeParam>(string(TEST_DIR"/translate/translate_large_1.test"), 1,
                             af::dim4(250, 320, 1, 1), -20, 24, AF_INTERP_NEAREST);
}

TYPED_TEST(Translate, Large3)
{
    translateTest<TypeParam>(string(TEST_DIR"/translate/translate_large_1.test"), 2,
                             af::dim4(300, 400, 1, 1), 10.23, 12.72, AF_INTERP_BILINEAR);
}

TYPED_TEST(Translate, Large4)
{
    translateTest<TypeParam>(string(TEST_DIR"/translate/translate_large_1.test"), 3,
                             af::dim4(300, 400, 1, 1), -15.69, -10.13, AF_INTERP_BILINEAR);
}

TYPED_TEST(TranslateInt, Small1)
{
    translateTest<TypeParam>(string(TEST_DIR"/translate/translate_small_1.test"), 0,
                             af::dim4(10, 10, 1, 1), 3, 2, AF_INTERP_NEAREST);
}

TYPED_TEST(TranslateInt, Small2)
{
    translateTest<TypeParam>(string(TEST_DIR"/translate/translate_small_1.test"), 1,
                             af::dim4(10, 10, 1, 1), -3, -2, AF_INTERP_NEAREST);
}

TYPED_TEST(TranslateInt, Small3)
{
    translateTest<TypeParam>(string(TEST_DIR"/translate/translate_small_1.test"), 2,
                             af::dim4(15, 15, 1, 1), 1.5, 2.5, AF_INTERP_BILINEAR);
}

TYPED_TEST(TranslateInt, Small4)
{
    translateTest<TypeParam>(string(TEST_DIR"/translate/translate_small_1.test"), 3,
                             af::dim4(15, 15, 1, 1), -1.5, -2.5, AF_INTERP_BILINEAR);
}

TYPED_TEST(TranslateInt, Large1)
{
    translateTest<TypeParam>(string(TEST_DIR"/translate/translate_large_1.test"), 0,
                             af::dim4(250, 320, 1, 1), 10, 18, AF_INTERP_NEAREST);
}

TYPED_TEST(TranslateInt, Large2)
{
    translateTest<TypeParam>(string(TEST_DIR"/translate/translate_large_1.test"), 1,
                             af::dim4(250, 320, 1, 1), -20, 24, AF_INTERP_NEAREST);
}

TYPED_TEST(TranslateInt, Large3)
{
    translateTest<TypeParam>(string(TEST_DIR"/translate/translate_large_1.test"), 2,
                             af::dim4(300, 400, 1, 1), 10.23, 12.72, AF_INTERP_BILINEAR, 0.001);
}

TYPED_TEST(TranslateInt, Large4)
{
    translateTest<TypeParam>(string(TEST_DIR"/translate/translate_large_1.test"), 3,
                             af::dim4(300, 400, 1, 1), -15.69, -10.13, AF_INTERP_BILINEAR, 0.001);
}
