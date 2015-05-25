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
#include <testHelpers.hpp>

using std::string;
using std::vector;

template<typename T>
class Sobel : public ::testing::Test
{
    public:
        virtual void SetUp() {}
};

template<typename T>
class Sobel_Integer : public ::testing::Test
{
    public:
        virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double> TestTypes;
typedef ::testing::Types<int, unsigned, char, unsigned char> TestTypesInt;

// register the type list
TYPED_TEST_CASE(Sobel, TestTypes);
TYPED_TEST_CASE(Sobel_Integer, TestTypesInt);

template<typename Ti, typename To>
void testSobelDerivatives(string pTestFile)
{
    if (noDoubleTests<Ti>()) return;

    vector<af::dim4>  numDims;
    vector<vector<Ti> >      in;
    vector<vector<To> >   tests;

    readTests<Ti,To,int>(pTestFile, numDims, in, tests);

    af::dim4 dims    = numDims[0];
    af_array dxArray = 0;
    af_array dyArray = 0;
    af_array inArray = 0;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in[0].front()),
                dims.ndims(), dims.get(), (af_dtype)af::dtype_traits<Ti>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_sobel_operator(&dxArray, &dyArray, inArray, 3));

    To *dxData = new To[dims.elements()];
    To *dyData = new To[dims.elements()];

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)dxData, dxArray));
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)dyData, dyArray));

    vector<To> currDXGoldBar = tests[0];
    vector<To> currDYGoldBar = tests[1];
    size_t nElems = currDXGoldBar.size();
    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(currDXGoldBar[elIter], dxData[elIter])<< "at: " << elIter<< std::endl;
    }
    nElems = currDYGoldBar.size();
    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(currDYGoldBar[elIter], dyData[elIter])<< "at: " << elIter<< std::endl;
    }

    // cleanup
    delete[] dxData;
    delete[] dyData;
    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(dxArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(dyArray));
}

TYPED_TEST(Sobel, Rectangle)
{
    testSobelDerivatives<TypeParam, TypeParam>(string(TEST_DIR"/sobel/rectangle.test"));
}

TYPED_TEST(Sobel_Integer, Rectangle)
{
    testSobelDerivatives<TypeParam, int>(string(TEST_DIR"/sobel/rectangle.test"));
}
