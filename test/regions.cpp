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
using af::cfloat;
using af::cdouble;

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
void regionsTest(string pTestFile, af_connectivity connectivity, bool isSubRef = false, const vector<af_seq> * seqv = NULL)
{
    if (noDoubleTests<T>()) return;

    vector<af::dim4> numDims;
    vector<vector<uchar> > in;
    vector<vector<T> > tests;
    readTests<uchar, T, unsigned>(pTestFile,numDims,in,tests);

    af::dim4 idims = numDims[0];

    af_array inArray = 0;
    af_array tempArray = 0;
    af_array outArray = 0;

    if (isSubRef) {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&tempArray, &(in[0].front()), idims.ndims(), idims.get(), (af_dtype) af::dtype_traits<char>::af_type));

        ASSERT_EQ(AF_SUCCESS, af_index(&inArray, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in[0].front()), idims.ndims(), idims.get(), (af_dtype) af::dtype_traits<char>::af_type));
    }

    ASSERT_EQ(AF_SUCCESS, af_regions(&outArray, inArray, connectivity, (af_dtype) af::dtype_traits<T>::af_type));

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

    if(inArray   != 0) af_release_array(inArray);
    if(outArray  != 0) af_release_array(outArray);
    if(tempArray != 0) af_release_array(tempArray);
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
    if (noDoubleTests<float>()) return;

    vector<af::dim4> numDims;
    vector<vector<float> > in;
    vector<vector<float> > tests;
    readTests<float, float, unsigned>(string(TEST_DIR"/regions/regions_8x8_4.test"),numDims,in,tests);

    af::dim4 idims = numDims[0];
    af::array input(idims, (float*)&(in[0].front()));
    af::array output = af::regions(input.as(b8));

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

///////////////////////////////// Documentation Examples ///////////////////
TEST(Regions, Docs_8)
{
    // input data
    uchar input[64] =  {
        0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 1, 0, 1, 0, 0, 1,
        0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 1, 0, 0,
        1, 0, 0, 1, 0, 0, 1, 0,
        0, 0, 0, 1, 1, 0, 0, 1,
        1, 1, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 1, 1, 1, 1, 0
    };
    // gold output
    float gold[64] =  {
        0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 1, 0, 1, 0, 0, 2,
        0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 3, 0, 0,
        4, 0, 0, 1, 0, 0, 3, 0,
        0, 0, 0, 1, 1, 0, 0, 3,
        5, 5, 0, 0, 0, 0, 0, 0,
        0, 5, 0, 6, 6, 6, 6, 0
    };

    //![ex_image_regions]
    af::array in(8, 8, input);
    //af_print(in);
    // in =
    // 0   0   0   0   1   0   1   0
    // 0   0   0   0   0   0   1   1
    // 0   1   0   1   0   0   0   0
    // 0   0   1   0   1   1   0   1
    // 1   1   0   0   0   1   0   1
    // 0   0   0   1   0   0   0   1
    // 0   0   0   0   1   0   0   1
    // 0   1   0   0   0   1   0   0

    // Compute the label matrix using 8-way connectivity
    af::array out = regions(in.as(b8), AF_CONNECTIVITY_8);
    //af_print(out);
    // 0   0   0   0   4   0   5   0
    // 0   0   0   0   0   0   5   5
    // 0   1   0   1   0   0   0   0
    // 0   0   1   0   1   1   0   6
    // 1   1   0   0   0   1   0   6
    // 0   0   0   3   0   0   0   6
    // 0   0   0   0   3   0   0   6
    // 0   2   0   0   0   3   0   0
    //![ex_image_regions]


    float output[64];
    out.host((void*)output);

    for (int i=0; i<64; ++i) {
        ASSERT_EQ(gold[i], output[i])<<" mismatch at i="<<i<<std::endl;
    }
}

TEST(Regions, Docs_4)
{
    // input data
    uchar input[64] =  {
        0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 1, 0, 1, 0, 0, 1,
        0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 1, 0, 0,
        1, 0, 0, 1, 0, 0, 1, 0,
        0, 0, 0, 1, 1, 0, 0, 1,
        1, 1, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 1, 1, 1, 1, 0
    };
    // gold output
    float gold[64] =  {
        0.0000,  0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,
        0.0000,  0.0000,  2.0000,  0.0000,  1.0000,  0.0000,  0.0000,  3.0000,
        0.0000,  0.0000,  0.0000,  4.0000,  0.0000,  0.0000,  0.0000,  0.0000,
        0.0000,  0.0000,  5.0000,  0.0000,  0.0000,  6.0000,  0.0000,  0.0000,
        7.0000,  0.0000,  0.0000,  8.0000,  0.0000,  0.0000,  9.0000,  0.0000,
        0.0000,  0.0000,  0.0000,  8.0000,  8.0000,  0.0000,  0.0000, 10.0000,
        11.000, 11.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
        0.0000, 11.0000,  0.0000, 12.0000, 12.0000, 12.0000, 12.0000,  0.0000
    };


    //![ex_image_regions_4conn]
    af::array in(8, 8, input);
    //af_print(in.T());
    //in
    //0  0  0  0  1  0  1  0
    //0  0  0  0  0  0  1  1
    //0  1  0  1  0  0  0  0
    //0  0  1  0  1  1  0  1
    //1  1  0  0  0  1  0  1
    //0  0  0  1  0  0  0  1
    //0  0  0  0  1  0  0  1
    //0  1  0  0  0  1  0  0
    // Compute the label matrix using 4-way connectivity
    af::array out = regions(in.as(b8), AF_CONNECTIVITY_4);
    //af_print(out.T());
    //out
    //0  0  0  0  7  0 11  0
    //0  0  0  0  0  0 11 11
    //0  2  0  5  0  0  0  0
    //0  0  4  0  8  8  0 12
    //1  1  0  0  0  8  0 12
    //0  0  0  6  0  0  0 12
    //0  0  0  0  9  0  0 12
    //0  3  0  0  0 10  0  0
    //![ex_image_regions_4conn]


    float output[64];
    out.host((void*)output);

    for (int i=0; i<64; ++i) {
        ASSERT_EQ(gold[i], output[i])<<" mismatch at i="<<i<<std::endl;
    }
}
