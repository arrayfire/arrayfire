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

// create a list of types to be tested
typedef ::testing::Types<float, double, int, char> TestTypes;

// register the type list
TYPED_TEST_CASE(Sobel, TestTypes);

template<typename T>
void testSobelDerivatives(string pTestFile)
{
    if (noDoubleTests<T>()) return;

    vector<af::dim4>  numDims;
    vector<vector<T>>      in;
    vector<vector<T>>   tests;

    readTests<T,T,int>(pTestFile, numDims, in, tests);

    af::dim4 dims    = numDims[0];
    af_array dxArray = 0;
    af_array dyArray = 0;
    af_array inArray = 0;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in[0].front()),
                dims.ndims(), dims.get(), (af_dtype)af::dtype_traits<T>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_sobel_operator(&dxArray, &dyArray, inArray, 3));

    T *dxData = new T[dims.elements()];
    T *dyData = new T[dims.elements()];

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)dxData, dxArray));
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)dyData, dyArray));

    vector<T> currDXGoldBar = tests[0];
    vector<T> currDYGoldBar = tests[1];
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
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(inArray));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(dxArray));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(dyArray));
}

TYPED_TEST(Sobel, Rectangle)
{
    testSobelDerivatives<TypeParam>(string(TEST_DIR"/sobel/rectangle.test"));

}
