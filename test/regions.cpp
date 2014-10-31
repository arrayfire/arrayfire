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
#include <af/defines.h>
#include <af/traits.hpp>
#include <af/image.h>
#include <vector>
#include <iostream>
#include <string>
#include <testHelpers.hpp>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using af::af_cfloat;
using af::af_cdouble;

template<typename T>
class Regions : public ::testing::Test
{
    public:
        virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double, int, unsigned> TestTypes;

// register the type list
TYPED_TEST_CASE(Regions, TestTypes);

template<typename T>
void regionsTest(string pTestFile, af_connectivity_type connectivity, bool isSubRef = false, const vector<af_seq> * seqv = nullptr)
{
    vector<af::dim4> numDims;
    vector<vector<uchar>> in;
    vector<vector<T>> tests;
    readTests<uchar, T, unsigned>(pTestFile,numDims,in,tests);

    af::dim4 idims = numDims[0];

    af_array inArray = 0;
    af_array tempArray = 0;
    af_array outArray = 0;

    if (isSubRef) {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&tempArray, &(in[0].front()), idims.ndims(), idims.get(), (af_dtype) af::dtype_traits<T>::af_type));

        ASSERT_EQ(AF_SUCCESS, af_index(&inArray, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in[0].front()), idims.ndims(), idims.get(), (af_dtype) af::dtype_traits<T>::af_type));
    }

    ASSERT_EQ(AF_SUCCESS, af_regions(&outArray, inArray, connectivity));

    // Get result
    T* outData = new T[idims.elements()];
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    // Compare result
    for (size_t testIter = 0; testIter < tests.size(); ++testIter) {
        vector<T> currGoldBar = tests[testIter];
        size_t nElems = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter]) << "at: " << elIter << std::endl;
        }
    }

    // Delete
    delete[] outData;

    if(inArray   != 0) af_destroy_array(inArray);
    if(outArray  != 0) af_destroy_array(outArray);
    if(tempArray != 0) af_destroy_array(tempArray);
}

#define REGIONS_INIT(desc, file, conn, conn_type)                                           \
    TYPED_TEST(Regions, desc)                                                               \
    {                                                                                       \
        regionsTest<TypeParam>(string(TEST_DIR"/regions/"#file"_"#conn".test"), conn_type); \
    }

    REGIONS_INIT(Regions0, regions_8x8, 4, AF_CONNECTIVITY_4);
    REGIONS_INIT(Regions1, regions_8x8, 8, AF_CONNECTIVITY_8);
    REGIONS_INIT(Regions2, regions_128x128, 4, AF_CONNECTIVITY_4);
    REGIONS_INIT(Regions3, regions_128x128, 8, AF_CONNECTIVITY_8);


///////////////////////////////////// CPP ////////////////////////////////
//
TEST(Regions, CPP)
{
    vector<af::dim4> numDims;
    vector<vector<uchar>> in;
    vector<vector<float>> tests;
    readTests<uchar, float, unsigned>(string(TEST_DIR"/regions/regions_8x8_4.test"),numDims,in,tests);

    af::dim4 idims = numDims[0];
    af::array input(idims, (float*)&(in[0].front()));
    af::array output = af::regions(input);

    // Get result
    float* outData = new float[idims.elements()];
    output.host((void*)outData);

    // Compare result
    for (size_t testIter = 0; testIter < tests.size(); ++testIter) {
        vector<float> currGoldBar = tests[testIter];
        size_t nElems = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter]) << "at: " << elIter << std::endl;
        }
    }

    // Delete
    delete[] outData;
}
